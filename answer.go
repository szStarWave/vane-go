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
	searchResults, err := gatherSearchResults(ctx, provider, query, mode, req.OnSearchEvent)
	if err != nil || len(searchResults) == 0 {
		if err != nil {
			notifySearchError(req.OnSearchError, err)
			emitSearchEvent(ctx, req.OnSearchEvent, SearchEvent{
				Type:  SearchEventError,
				Mode:  mode,
				Query: query,
				Error: err.Error(),
			})
		}
		return req.Model.GenerateContent(ctx, &model.Request{
			Messages:         req.Messages,
			GenerationConfig: withStreaming(req.GenerationConfig),
			ExtraFields:      req.ExtraFields,
		})
	}
	now := req.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	genConfig := withStreaming(req.GenerationConfig)
	if genConfig.Temperature == nil {
		genConfig.Temperature = floatPtr(0.4)
	}
	writerReq := &model.Request{
		Messages:         buildWriterMessages(req.Messages, searchResults, req.SystemInstructions, mode, now),
		GenerationConfig: genConfig,
		ExtraFields:      req.ExtraFields,
	}
	return req.Model.GenerateContent(ctx, writerReq)
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

func buildWriterMessages(messages []model.Message, results []SearchResult, instructions string, mode Mode, now time.Time) []model.Message {
	writerPrompt := getWriterPrompt(formatSearchContext(results), instructions, mode, now)
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
	for i, result := range results {
		title := xmlishEscape(result.Title)
		content := xmlishEscape(result.Content)
		resultURL := xmlishEscape(result.URL)
		fmt.Fprintf(&b, `<result index="%d" title="%s" url="%s">%s</result>`+"\n", i+1, title, resultURL, content)
	}
	b.WriteString("</search_results>")
	return b.String()
}

func getWriterPrompt(contextText, systemInstructions string, mode Mode, now time.Time) string {
	depth := ""
	if mode == ModeQuality {
		depth = "\n- You are in quality search mode: provide a deeper, more comprehensive answer when the sources support it."
	}
	if mode == ModeSpeed {
		depth = "\n- You are in speed search mode: answer concisely and prioritize the most relevant source facts."
	}
	return fmt.Sprintf(`You are Vane, an AI answering engine skilled in web search and source-grounded answers.

Your task:
- Answer the user's latest question using the provided web context.
- Use Markdown for clear structure.
- Cite source-backed claims with [number] notation matching the result index.
- If the context is insufficient, say what is missing instead of inventing facts.
- Do not cite facts that are not supported by the context.%s

User instructions:
%s

Current date and time (UTC): %s

<context>
%s
</context>`, depth, strings.TrimSpace(systemInstructions), now.Format(time.RFC3339), contextText)
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
