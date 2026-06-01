package vane

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

func Answer(ctx context.Context, req Request) (<-chan *model.Response, error) {
	if req.Model == nil {
		return nil, fmt.Errorf("vane: model is nil")
	}
	provider := req.SearchProvider
	if provider == nil {
		defaultProvider, err := NewWebsurfxProvider(nil)
		if err != nil {
			notifySearchError(req.OnSearchError, err)
			return fallbackGenerate(ctx, req.Model, req.Messages, nil), nil
		}
		provider = defaultProvider
	}
	query := latestUserText(req.Messages)
	if query == "" {
		return req.Model.GenerateContent(ctx, &model.Request{
			Messages:         req.Messages,
			GenerationConfig: withStreaming(req.GenerationConfig),
			ExtraFields:      req.ExtraFields,
		})
	}
	mode := req.Mode
	if mode == "" {
		mode = ModeBalanced
	}
	now := req.Now
	if now.IsZero() {
		now = time.Now()
	}
	agent := SearchAgent{
		ClassifierModel:       req.Model,
		ResearchModel:         firstModel(req.ResearchModel, req.Model),
		SearchProvider:        provider,
		ScrapeProvider:        req.ScrapeProvider,
		EmbeddingProvider:     req.EmbeddingProvider,
		TextEmbeddingProvider: req.TextEmbeddingProvider,
		WidgetProviders:       req.WidgetProviders,
		OnSearchEvent:         req.OnSearchEvent,
	}
	result, err := agent.Run(ctx, SearchAgentRequest{
		Query:         query,
		Messages:      req.Messages,
		Mode:          mode,
		Sources:       req.Sources,
		FileIDs:       req.FileIDs,
		MaxIterations: req.MaxIterations,
		Now:           now,
	})
	if err != nil {
		notifySearchError(req.OnSearchError, err)
		emitSearchEvent(ctx, req.OnSearchEvent, SearchEvent{
			Type:  SearchEventError,
			Mode:  mode,
			Query: query,
			Error: err.Error(),
		})
	}
	if !result.Classification.ShouldSearch && len(result.Sources) == 0 && len(result.Widgets) == 0 {
		if err != nil {
			return fallbackGenerate(ctx, req.Model, req.Messages, &req.GenerationConfig), nil
		}
		return req.Model.GenerateContent(ctx, &model.Request{
			Messages:         req.Messages,
			GenerationConfig: withStreaming(req.GenerationConfig),
			ExtraFields:      req.ExtraFields,
		})
	}
	genConfig := withStreaming(req.GenerationConfig)
	if genConfig.Temperature == nil {
		genConfig.Temperature = floatPtr(0.4)
	}
	emitSearchEvent(ctx, req.OnSearchEvent, SearchEvent{
		Type:        SearchEventWriterStart,
		Mode:        mode,
		Query:       query,
		ResultCount: len(result.Sources),
	})
	writerReq := &model.Request{
		Messages:         buildWriterMessages(req.Messages, result.Sources, result.Widgets, req.SystemInstructions, mode, now),
		GenerationConfig: genConfig,
		ExtraFields:      req.ExtraFields,
	}
	ch, err := req.Model.GenerateContent(ctx, writerReq)
	if err != nil {
		emitSearchEvent(ctx, req.OnSearchEvent, SearchEvent{
			Type:  SearchEventError,
			Mode:  mode,
			Query: query,
			Error: err.Error(),
		})
		return nil, err
	}
	return wrapWriterEvents(ctx, ch, req.OnSearchEvent, mode, query), nil
}

func fallbackGenerate(ctx context.Context, m model.Model, messages []model.Message, cfg *model.GenerationConfig) <-chan *model.Response {
	req := &model.Request{Messages: messages, GenerationConfig: model.GenerationConfig{Stream: true}}
	if cfg != nil {
		req.GenerationConfig = *cfg
	}
	ch, err := m.GenerateContent(ctx, req)
	if err == nil {
		return ch
	}
	out := make(chan *model.Response, 1)
	out <- &model.Response{
		Object:    model.ObjectTypeChatCompletion,
		Timestamp: time.Now(),
		Done:      true,
		Error:     &model.ResponseError{Message: err.Error(), Type: model.ErrorTypeAPIError},
	}
	close(out)
	return out
}

func gatherSearchResults(ctx context.Context, provider SearchProvider, query string, mode Mode, eventHandler func(context.Context, SearchEvent)) ([]SearchResult, error) {
	queries := buildQueries(query, mode)
	limit := resultsPerQuery(mode)
	seen := map[string]bool{}
	var out []SearchResult
	var firstErr error
	emitSearchEvent(ctx, eventHandler, SearchEvent{
		Type:       SearchEventStart,
		Mode:       mode,
		Query:      query,
		Queries:    append([]string(nil), queries...),
		QueryTotal: len(queries),
	})
	for i, q := range queries {
		emitSearchEvent(ctx, eventHandler, SearchEvent{
			Type:       SearchEventQuery,
			Mode:       mode,
			Query:      q,
			QueryIndex: i + 1,
			QueryTotal: len(queries),
		})
		results, err := provider.Search(ctx, q, SearchOptions{Page: 1, Limit: limit})
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			emitSearchEvent(ctx, eventHandler, SearchEvent{
				Type:       SearchEventError,
				Mode:       mode,
				Query:      q,
				QueryIndex: i + 1,
				QueryTotal: len(queries),
				Error:      err.Error(),
			})
			continue
		}
		before := len(out)
		for _, result := range results {
			key := canonicalResultKey(result)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, result)
			if len(out) >= totalResultLimit(mode) {
				emitSearchEvent(ctx, eventHandler, SearchEvent{
					Type:        SearchEventResults,
					Mode:        mode,
					Query:       q,
					QueryIndex:  i + 1,
					QueryTotal:  len(queries),
					ResultCount: len(out),
					Results:     append([]SearchResult(nil), out[before:]...),
				})
				emitSearchEvent(ctx, eventHandler, SearchEvent{
					Type:        SearchEventEnd,
					Mode:        mode,
					Query:       query,
					Queries:     append([]string(nil), queries...),
					ResultCount: len(out),
					Results:     append([]SearchResult(nil), out...),
				})
				return out, nil
			}
		}
		emitSearchEvent(ctx, eventHandler, SearchEvent{
			Type:        SearchEventResults,
			Mode:        mode,
			Query:       q,
			QueryIndex:  i + 1,
			QueryTotal:  len(queries),
			ResultCount: len(out),
			Results:     append([]SearchResult(nil), out[before:]...),
		})
	}
	emitSearchEvent(ctx, eventHandler, SearchEvent{
		Type:        SearchEventEnd,
		Mode:        mode,
		Query:       query,
		Queries:     append([]string(nil), queries...),
		ResultCount: len(out),
		Results:     append([]SearchResult(nil), out...),
	})
	return out, firstErr
}

func buildQueries(query string, mode Mode) []string {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}
	switch mode {
	case ModeSpeed:
		return []string{query}
	case ModeQuality:
		return uniqueStrings([]string{
			query,
			query + " latest",
			query + " overview",
			query + " analysis",
			query + " limitations",
		})
	default:
		return uniqueStrings([]string{
			query,
			query + " latest",
			query + " overview",
		})
	}
}

func resultsPerQuery(mode Mode) int {
	switch mode {
	case ModeSpeed:
		return 8
	case ModeQuality:
		return 10
	default:
		return 8
	}
}

func totalResultLimit(mode Mode) int {
	switch mode {
	case ModeSpeed:
		return 8
	case ModeQuality:
		return 24
	default:
		return 16
	}
}

func buildWriterMessages(messages []model.Message, results []SearchResult, widgets []WidgetResult, instructions string, mode Mode, now time.Time) []model.Message {
	writerPrompt := getWriterPrompt(formatSearchContext(results), formatWidgetContext(widgets), instructions, mode, now)
	out := make([]model.Message, 0, len(messages)+1)
	out = append(out, model.Message{Role: model.RoleSystem, Content: writerPrompt})
	for _, msg := range messages {
		if msg.Role == model.RoleSystem {
			continue
		}
		out = append(out, msg)
	}
	return out
}

func formatSearchContext(results []SearchResult) string {
	var b strings.Builder
	b.WriteString(`<search_results note="These are web search results. Cite them with [number] notation.">`)
	b.WriteString("\n")
	if len(results) == 0 {
		b.WriteString(`<no_results>Search was attempted, but no usable source results were retrieved. Do not answer live or current factual claims as known facts without sources.</no_results>`)
		b.WriteString("\n")
	}
	for i, result := range results {
		title := xmlishEscape(result.Title)
		content := xmlishEscape(result.Content)
		resultURL := xmlishEscape(result.URL)
		fmt.Fprintf(&b, `<result index="%d" title="%s" url="%s">%s</result>`+"\n", i+1, title, resultURL, content)
	}
	b.WriteString("</search_results>")
	return b.String()
}

func formatWidgetContext(widgets []WidgetResult) string {
	if len(widgets) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString(`<widgets note="Structured widget outputs. Use when relevant and do not cite as web sources unless also supported by search results.">`)
	b.WriteString("\n")
	for i, widget := range widgets {
		fmt.Fprintf(&b, `<widget index="%d" kind="%s" title="%s">%s</widget>`+"\n", i+1, xmlishEscape(string(widget.Kind)), xmlishEscape(widget.Title), xmlishEscape(firstNonEmpty(widget.Content, widget.Error)))
	}
	b.WriteString("</widgets>")
	return b.String()
}

func getWriterPrompt(contextText, widgetText, systemInstructions string, mode Mode, now time.Time) string {
	depth := ""
	if mode == ModeQuality {
		depth = "\n- YOU ARE CURRENTLY SET IN QUALITY MODE: generate a very deep, detailed, comprehensive response using the full context provided. When the sources support it, frame the answer like a research report and aim for at least 2000 words."
	}
	if mode == ModeSpeed {
		depth = "\n- Speed mode: answer concisely and prioritize the most relevant source facts."
	}
	return fmt.Sprintf(`You are Vane, an AI model skilled in web search and crafting detailed, engaging, well-structured answers. You excel at summarizing web pages, extracting relevant information, and creating professional, blog-style responses.

Your task:
- Thoroughly answer the user's latest question using the provided research context.
- Use Markdown with clear headings and subheadings when useful.
- Maintain a neutral, journalistic tone with engaging narrative flow.
- Start directly with the answer or introduction; do not add a main title unless the user asks for one.
- Cite every source-backed fact, detail, sentence, or clause with [number] notation matching the search result index.
- Every sentence in the response that relies on web context should include at least one citation.
- Use multiple citations for the same detail when multiple sources support it, for example [1][2].
- If the context is insufficient, say what is missing instead of inventing facts.
- If search_results contains no result entries, clearly say the search did not retrieve usable sources and do not claim current events from memory.
- Use the current date below when interpreting relative dates; do not call dates before or equal to the current date "future".
- Do not cite facts that are not supported by the context.
- Widget outputs can inform the answer but must not be cited as web sources unless the same fact is supported by search_results.
- Prefer direct, useful answers over explaining the search process.
- Include a brief conclusion or synthesis when appropriate.%s

User instructions:
%s

Current date and time: %s

<context>
%s
%s
</context>`, depth, strings.TrimSpace(systemInstructions), now.Format(time.RFC3339), contextText, widgetText)
}

func latestUserText(messages []model.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != model.RoleUser {
			continue
		}
		if text := messageText(messages[i]); text != "" {
			return text
		}
	}
	return ""
}

func messageText(msg model.Message) string {
	if strings.TrimSpace(msg.Content) != "" {
		return strings.TrimSpace(msg.Content)
	}
	var parts []string
	for _, part := range msg.ContentParts {
		if part.Type == model.ContentTypeText && part.Text != nil {
			parts = append(parts, *part.Text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func wrapWriterEvents(ctx context.Context, in <-chan *model.Response, handler func(context.Context, SearchEvent), mode Mode, query string) <-chan *model.Response {
	if handler == nil {
		return in
	}
	out := make(chan *model.Response)
	go func() {
		defer close(out)
		for rsp := range in {
			if text := responseText(rsp); text != "" {
				emitSearchEvent(ctx, handler, SearchEvent{
					Type:    SearchEventWriterDelta,
					Mode:    mode,
					Query:   query,
					Message: text,
				})
			}
			if rsp != nil && rsp.Done {
				emitSearchEvent(ctx, handler, SearchEvent{
					Type:  SearchEventWriterEnd,
					Mode:  mode,
					Query: query,
				})
			}
			out <- rsp
		}
	}()
	return out
}

func responseText(rsp *model.Response) string {
	if rsp == nil {
		return ""
	}
	var parts []string
	for _, choice := range rsp.Choices {
		if strings.TrimSpace(choice.Message.Content) != "" {
			parts = append(parts, choice.Message.Content)
		}
	}
	return strings.TrimSpace(strings.Join(parts, ""))
}

func canonicalResultKey(result SearchResult) string {
	raw := strings.TrimSpace(result.URL)
	if raw == "" {
		return strings.ToLower(strings.TrimSpace(result.Title + "\n" + result.Content))
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return strings.ToLower(raw)
	}
	parsed.Fragment = ""
	return strings.ToLower(parsed.String())
}

func uniqueStrings(values []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		key := strings.ToLower(value)
		if value == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, value)
	}
	return out
}

func xmlishEscape(value string) string {
	value = strings.ReplaceAll(value, "&", "&amp;")
	value = strings.ReplaceAll(value, "<", "&lt;")
	value = strings.ReplaceAll(value, ">", "&gt;")
	value = strings.ReplaceAll(value, `"`, "&quot;")
	return value
}

func floatPtr(value float64) *float64 {
	return &value
}

func intPtr(value int) *int {
	return &value
}

func withStreaming(config model.GenerationConfig) model.GenerationConfig {
	config.Stream = true
	return config
}

func notifySearchError(handler func(error), err error) {
	if handler != nil && err != nil {
		handler(err)
	}
}

func emitSearchEvent(ctx context.Context, handler func(context.Context, SearchEvent), ev SearchEvent) {
	if handler != nil {
		handler(ctx, ev)
	}
}
