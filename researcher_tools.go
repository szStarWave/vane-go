package vane

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/tool"
	"trpc.group/trpc-go/trpc-agent-go/tool/function"
)

type researchActionResult struct {
	Type      string         `json:"type"`
	Reasoning string         `json:"reasoning,omitempty"`
	Results   []SearchResult `json:"results,omitempty"`
	Done      bool           `json:"done,omitempty"`
	Error     string         `json:"error,omitempty"`
}

type queryActionArgs struct {
	Queries []string `json:"queries"`
}

type planActionArgs struct {
	Plan string `json:"plan"`
}

type scrapeURLArgs struct {
	URLs []string `json:"urls"`
}

func (r Researcher) researchWithTools(ctx context.Context, req ResearchRequest) ([]SearchResult, bool, error) {
	iterations := req.MaxIterations
	if iterations <= 0 {
		iterations = defaultMaxIterations(req.Mode)
	}
	tools := r.availableResearchTools(req)
	if len(tools) == 0 {
		return nil, false, nil
	}
	query := firstNonEmpty(strings.TrimSpace(req.Classification.StandaloneFollowUp), req.Query)
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:       SearchEventStart,
		Mode:       req.Mode,
		Query:      query,
		QueryTotal: iterations,
	})
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:      SearchEventResearchStart,
		Mode:      req.Mode,
		Query:     query,
		StepTotal: iterations,
	})
	history := []model.Message{
		model.NewUserMessage(fmt.Sprintf("<conversation>\n%s\nUser: %s (Standalone question: %s)\n</conversation>", formatMessagesForPrompt(tailMessages(req.Messages, 10)), req.Query, query)),
	}
	var out []SearchResult
	seen := map[string]bool{}
	var firstErr error
	usedTools := false
	for step := 0; step < iterations; step++ {
		prompt := getResearcherPrompt(req, step, iterations)
		ch, err := r.ResearchModel.GenerateContent(ctx, &model.Request{
			Messages: append([]model.Message{model.NewSystemMessage(prompt)}, history...),
			GenerationConfig: model.GenerationConfig{
				Stream:      true,
				Temperature: floatPtr(0),
				MaxTokens:   intPtr(900),
			},
			Tools: tools,
		})
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			return nil, usedTools, firstErr
		}
		toolCalls, err := collectToolCalls(ctx, ch)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			return nil, usedTools, firstErr
		}
		if len(toolCalls) == 0 {
			return out, usedTools, firstErr
		}
		usedTools = true
		history = append(history, model.Message{Role: model.RoleAssistant, ToolCalls: toolCalls})
		done := false
		for _, call := range toolCalls {
			name := call.Function.Name
			args := callArgsMap(call.Function.Arguments)
			emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
				Type:   SearchEventToolCall,
				Mode:   req.Mode,
				Query:  query,
				Action: name,
				Step:   step + 1,
				ToolCall: &ToolCallEvent{
					ID:   firstNonEmpty(call.ID, fmt.Sprintf("%s-%d", name, step+1)),
					Name: name,
					Args: args,
					Step: step + 1,
				},
			})
			actionResult, err := r.executeResearchTool(ctx, req, name, call.Function.Arguments, step+1)
			if err != nil && firstErr == nil {
				firstErr = err
			}
			if actionResult.Type == "reasoning" && strings.TrimSpace(actionResult.Reasoning) != "" {
				emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
					Type:      SearchEventResearchStep,
					Mode:      req.Mode,
					Query:     query,
					Step:      step + 1,
					StepTotal: iterations,
					Message:   actionResult.Reasoning,
				})
			}
			before := len(out)
			for _, result := range actionResult.Results {
				key := canonicalResultKey(result)
				if key == "" || seen[key] {
					continue
				}
				seen[key] = true
				out = append(out, result)
			}
			if len(out) > totalResultLimit(req.Mode) {
				out = out[:totalResultLimit(req.Mode)]
			}
			resultEvent := &ToolResultEvent{
				ID:          firstNonEmpty(call.ID, fmt.Sprintf("%s-%d", name, step+1)),
				Name:        name,
				ResultCount: len(actionResult.Results),
				Results:     actionResult.Results,
				Error:       firstNonEmpty(actionResult.Error, errorString(err)),
				Step:        step + 1,
			}
			emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
				Type:        SearchEventToolResult,
				Mode:        req.Mode,
				Query:       query,
				Action:      name,
				Step:        step + 1,
				ToolResult:  resultEvent,
				ResultCount: len(actionResult.Results),
				Results:     actionResult.Results,
			})
			if len(out) > before {
				emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
					Type:        SearchEventResults,
					Mode:        req.Mode,
					Query:       query,
					QueryIndex:  step + 1,
					QueryTotal:  iterations,
					ResultCount: len(out),
					Results:     append([]SearchResult(nil), out[before:]...),
				})
			}
			payload, _ := json.Marshal(actionResult)
			history = append(history, model.NewToolMessage(firstNonEmpty(call.ID, fmt.Sprintf("%s-%d", name, step+1)), name, string(payload)))
			if actionResult.Done || name == "done" {
				done = true
			}
		}
		if done || len(out) >= totalResultLimit(req.Mode) {
			break
		}
	}
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:        SearchEventEnd,
		Mode:        req.Mode,
		Query:       query,
		ResultCount: len(out),
		Results:     append([]SearchResult(nil), out...),
	})
	return out, usedTools, firstErr
}

func (r Researcher) availableResearchTools(req ResearchRequest) map[string]tool.Tool {
	tools := map[string]tool.Tool{}
	if req.Mode != ModeSpeed {
		tools["__reasoning_preamble"] = function.NewFunctionTool(
			func(_ context.Context, args planActionArgs) (researchActionResult, error) {
				return researchActionResult{Type: "reasoning", Reasoning: args.Plan}, nil
			},
			function.WithName("__reasoning_preamble"),
			function.WithDescription("Use this FIRST and before every non-done tool call to state a concise natural-language research plan."),
			function.WithInputSchema(&tool.Schema{Type: "object", Required: []string{"plan"}, Properties: map[string]*tool.Schema{
				"plan": {Type: "string", Description: "Concise natural-language plan."},
			}}),
		)
	}
	if hasSource(req.Sources, SearchSourceWeb) && !req.Classification.SkipSearch {
		tools["web_search"] = r.queryTool("web_search", SearchSourceWeb, "Perform web searches for concise targeted queries.")
	}
	if hasSource(req.Sources, SearchSourceAcademic) && req.Classification.AcademicSearch && !req.Classification.SkipSearch {
		tools["academic_search"] = r.queryTool("academic_search", SearchSourceAcademic, "Search scholarly, paper, benchmark, official report, and research-oriented sources.")
	}
	if hasSource(req.Sources, SearchSourceDiscussions) && req.Classification.DiscussionSearch && !req.Classification.SkipSearch {
		tools["social_search"] = r.queryTool("social_search", SearchSourceDiscussions, "Search community discussions, issues, forums, and social feedback.")
	}
	if len(req.FileIDs) > 0 && r.EmbeddingProvider != nil {
		tools["uploads_search"] = r.queryTool("uploads_search", SearchSourceUploads, "Search the user's uploaded files and personal documents.")
	}
	tools["scrape_url"] = function.NewFunctionTool(
		func(ctx context.Context, args scrapeURLArgs) (researchActionResult, error) {
			return r.scrapeURLs(ctx, req, args.URLs)
		},
		function.WithName("scrape_url"),
		function.WithDescription("Scrape and extract content from specific URLs. Use only when the user asks about exact URLs or when deep reading is necessary."),
		function.WithInputSchema(&tool.Schema{Type: "object", Required: []string{"urls"}, Properties: map[string]*tool.Schema{
			"urls": {Type: "array", Description: "URLs to scrape, maximum 3.", Items: &tool.Schema{Type: "string"}},
		}}),
	)
	tools["done"] = function.NewFunctionTool(
		func(context.Context, struct{}) (researchActionResult, error) {
			return researchActionResult{Type: "done", Done: true}, nil
		},
		function.WithName("done"),
		function.WithDescription("Call only when research is complete and enough information has been gathered."),
		function.WithInputSchema(&tool.Schema{Type: "object", AdditionalProperties: false}),
	)
	return tools
}

func (r Researcher) queryTool(name string, source SearchSource, description string) tool.Tool {
	return function.NewFunctionTool(
		func(ctx context.Context, args queryActionArgs) (researchActionResult, error) {
			results, err := r.executeQueries(ctx, currentResearchRequest(ctx), source, args.Queries)
			res := researchActionResult{Type: "search_results", Results: results}
			if err != nil {
				res.Error = err.Error()
			}
			return res, err
		},
		function.WithName(name),
		function.WithDescription(description+" Provide up to 3 SEO-friendly keyword queries."),
		function.WithInputSchema(&tool.Schema{Type: "object", Required: []string{"queries"}, Properties: map[string]*tool.Schema{
			"queries": {Type: "array", Description: "Search queries, maximum 3.", Items: &tool.Schema{Type: "string"}},
		}}),
	)
}

type researchRequestContextKey struct{}

func withResearchRequest(ctx context.Context, req ResearchRequest) context.Context {
	return context.WithValue(ctx, researchRequestContextKey{}, req)
}

func currentResearchRequest(ctx context.Context) ResearchRequest {
	req, _ := ctx.Value(researchRequestContextKey{}).(ResearchRequest)
	return req
}

func (r Researcher) executeResearchTool(ctx context.Context, req ResearchRequest, name string, args []byte, step int) (researchActionResult, error) {
	ctx = withResearchRequest(ctx, req)
	switch name {
	case "__reasoning_preamble":
		var parsed planActionArgs
		_ = json.Unmarshal(args, &parsed)
		return researchActionResult{Type: "reasoning", Reasoning: parsed.Plan}, nil
	case "web_search":
		var parsed queryActionArgs
		_ = json.Unmarshal(args, &parsed)
		results, err := r.executeQueries(ctx, req, SearchSourceWeb, parsed.Queries)
		return actionResultFromSearch(results, err), err
	case "academic_search":
		var parsed queryActionArgs
		_ = json.Unmarshal(args, &parsed)
		results, err := r.executeQueries(ctx, req, SearchSourceAcademic, parsed.Queries)
		return actionResultFromSearch(results, err), err
	case "social_search":
		var parsed queryActionArgs
		_ = json.Unmarshal(args, &parsed)
		results, err := r.executeQueries(ctx, req, SearchSourceDiscussions, parsed.Queries)
		return actionResultFromSearch(results, err), err
	case "uploads_search":
		var parsed queryActionArgs
		_ = json.Unmarshal(args, &parsed)
		results, err := r.executeQueries(ctx, req, SearchSourceUploads, parsed.Queries)
		return actionResultFromSearch(results, err), err
	case "scrape_url":
		var parsed scrapeURLArgs
		_ = json.Unmarshal(args, &parsed)
		return r.scrapeURLs(ctx, req, parsed.URLs)
	case "done":
		return researchActionResult{Type: "done", Done: true}, nil
	default:
		err := fmt.Errorf("vane: unknown research action %s", name)
		return researchActionResult{Type: "error", Error: err.Error()}, err
	}
}

func actionResultFromSearch(results []SearchResult, err error) researchActionResult {
	out := researchActionResult{Type: "search_results", Results: results}
	if err != nil {
		out.Error = err.Error()
	}
	return out
}

func (r Researcher) executeQueries(ctx context.Context, req ResearchRequest, source SearchSource, queries []string) ([]SearchResult, error) {
	if len(queries) == 0 {
		queries = []string{firstNonEmpty(req.Classification.StandaloneFollowUp, req.Query)}
	}
	queries = uniqueStrings(queries)
	if len(queries) > 3 {
		queries = queries[:3]
	}
	var all []SearchResult
	var firstErr error
	for _, q := range queries {
		q = resolveRelativeDateQuery(q, req.Now)
		results, err := r.runAction(ctx, req, source, q)
		if err != nil && firstErr == nil {
			firstErr = err
		}
		for i := range results {
			if results[i].Source == "" {
				results[i].Source = source
			}
		}
		all = append(all, results...)
	}
	all = r.rankAndDedupe(ctx, queries, all, req.Mode)
	if req.Mode == ModeQuality && source != SearchSourceUploads {
		all = r.deepReadQuality(ctx, req, queries, all)
	}
	if len(all) > totalResultLimit(req.Mode) {
		all = all[:totalResultLimit(req.Mode)]
	}
	return all, firstErr
}

func (r Researcher) rankAndDedupe(ctx context.Context, queries []string, results []SearchResult, mode Mode) []SearchResult {
	if len(results) == 0 {
		return nil
	}
	if r.TextEmbeddingProvider == nil || mode == ModeQuality {
		return dedupeResults(results)
	}
	texts := append([]string{strings.Join(queries, "\n")}, resultTexts(results)...)
	embeddings, err := r.TextEmbeddingProvider.EmbedTexts(ctx, texts)
	if err != nil || len(embeddings) != len(texts) {
		return dedupeResults(results)
	}
	queryEmbedding := embeddings[0]
	type scored struct {
		result SearchResult
		score  float64
	}
	var scoredResults []scored
	for i, result := range results {
		score := cosineSimilarity(queryEmbedding, embeddings[i+1])
		if score < 0.25 {
			continue
		}
		scoredResults = append(scoredResults, scored{result: result, score: score})
	}
	sort.SliceStable(scoredResults, func(i, j int) bool { return scoredResults[i].score > scoredResults[j].score })
	ranked := make([]SearchResult, 0, len(scoredResults))
	for _, item := range scoredResults {
		ranked = append(ranked, item.result)
	}
	return dedupeResults(ranked)
}

func (r Researcher) deepReadQuality(ctx context.Context, req ResearchRequest, queries []string, results []SearchResult) []SearchResult {
	if r.ScrapeProvider == nil || len(results) == 0 {
		return results
	}
	picked := r.pickQualityResults(ctx, queries, results)
	if len(picked) == 0 {
		picked = results
	}
	if len(picked) > 3 {
		picked = picked[:3]
	}
	var out []SearchResult
	for _, result := range picked {
		if strings.TrimSpace(result.URL) == "" {
			out = append(out, result)
			continue
		}
		doc, err := r.ScrapeProvider.Scrape(ctx, result.URL)
		if err != nil || strings.TrimSpace(doc.Content) == "" {
			out = append(out, result)
			continue
		}
		facts := r.extractFacts(ctx, queries, doc.Content)
		if strings.TrimSpace(facts) == "" {
			facts = doc.Content
		}
		out = append(out, SearchResult{
			Title:   firstNonEmpty(doc.Title, result.Title),
			URL:     firstNonEmpty(doc.URL, result.URL),
			Content: facts,
			Source:  result.Source,
		})
	}
	if len(out) == 0 {
		return results
	}
	return out
}

func (r Researcher) pickQualityResults(ctx context.Context, queries []string, results []SearchResult) []SearchResult {
	if r.ResearchModel == nil || len(results) <= 3 {
		return results
	}
	var b strings.Builder
	for i, result := range results {
		fmt.Fprintf(&b, `<result index="%d" title="%s" url="%s">%s</result>`+"\n", i, xmlishEscape(result.Title), xmlishEscape(result.URL), xmlishEscape(result.Content))
	}
	ch, err := r.ResearchModel.GenerateContent(ctx, &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage("You are Vane's search result picker. Return compact JSON only: {\"picked_indices\":[0,1,2]}. Pick at most 3 relevant, credible, diverse results worth scraping."),
			model.NewUserMessage(fmt.Sprintf("<queries>%s</queries>\n<search_results>%s</search_results>", strings.Join(queries, ", "), b.String())),
		},
		GenerationConfig: model.GenerationConfig{Stream: false, Temperature: floatPtr(0), MaxTokens: intPtr(160)},
	})
	if err != nil {
		return results
	}
	text, err := collectResponseText(ctx, ch)
	if err != nil {
		return results
	}
	var parsed struct {
		PickedIndices []int `json:"picked_indices"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(text)), &parsed); err != nil {
		return results
	}
	var picked []SearchResult
	for _, index := range parsed.PickedIndices {
		if index >= 0 && index < len(results) {
			picked = append(picked, results[index])
		}
		if len(picked) >= 3 {
			break
		}
	}
	return picked
}

func (r Researcher) extractFacts(ctx context.Context, queries []string, content string) string {
	if r.ResearchModel == nil {
		return content
	}
	var out strings.Builder
	for _, chunk := range splitText(content, 4000, 500) {
		ch, err := r.ResearchModel.GenerateContent(ctx, &model.Request{
			Messages: []model.Message{
				model.NewSystemMessage("You are Vane's information extractor. Return compact JSON only: {\"extracted_facts\":\"- Fact\"}. Extract only facts relevant to the queries, preserve raw numbers and table values, remove boilerplate."),
				model.NewUserMessage(fmt.Sprintf("<queries>%s</queries>\n<scraped_data>%s</scraped_data>", strings.Join(queries, ", "), chunk)),
			},
			GenerationConfig: model.GenerationConfig{Stream: false, Temperature: floatPtr(0), MaxTokens: intPtr(900)},
		})
		if err != nil {
			continue
		}
		text, err := collectResponseText(ctx, ch)
		if err != nil {
			continue
		}
		var parsed struct {
			ExtractedFacts string `json:"extracted_facts"`
		}
		if err := json.Unmarshal([]byte(extractJSONObject(text)), &parsed); err == nil && strings.TrimSpace(parsed.ExtractedFacts) != "" {
			out.WriteString(parsed.ExtractedFacts)
			out.WriteString("\n")
		}
	}
	return strings.TrimSpace(out.String())
}

func (r Researcher) scrapeURLs(ctx context.Context, req ResearchRequest, urls []string) (researchActionResult, error) {
	if r.ScrapeProvider == nil {
		err := fmt.Errorf("vane: scrape provider is not configured")
		return researchActionResult{Type: "search_results", Error: err.Error()}, nil
	}
	urls = uniqueStrings(urls)
	if len(urls) > 3 {
		urls = urls[:3]
	}
	var results []SearchResult
	for _, rawURL := range urls {
		doc, err := r.ScrapeProvider.Scrape(ctx, rawURL)
		if err != nil {
			results = append(results, SearchResult{Title: "Error scraping " + rawURL, URL: rawURL, Content: err.Error()})
			continue
		}
		content := doc.Content
		if req.Mode == ModeQuality {
			if facts := r.extractFacts(ctx, []string{req.Query}, doc.Content); facts != "" {
				content = facts
			}
		}
		results = append(results, SearchResult{Title: firstNonEmpty(doc.Title, rawURL), URL: firstNonEmpty(doc.URL, rawURL), Content: content})
	}
	return researchActionResult{Type: "search_results", Results: results}, nil
}

func getResearcherPrompt(req ResearchRequest, i, maxIteration int) string {
	today := time.Now().Format("January 2, 2006")
	if !req.Now.IsZero() {
		today = req.Now.Format("January 2, 2006")
	}
	actionDesc := researchActionDescriptions(req)
	base := fmt.Sprintf(`Assistant is Vane's action orchestrator. Fulfill the user's request by selecting available tools only; do not write the final answer directly.

Today's date: %s
Research iteration: %d of %d

<available_tools>
%s
</available_tools>

<response_protocol>
- NEVER output normal text. ONLY call tools.
- Use targeted keyword queries, maximum 3 per search tool call.
- Default to web_search when information is missing or stale.
- Call done when enough information has been gathered.
- Do not invent tools.
</response_protocol>`, today, i+1, maxIteration, actionDesc)
	switch req.Mode {
	case ModeSpeed:
		return base + "\nSpeed mode: act quickly. One strong search call followed by done is usually enough."
	case ModeQuality:
		return base + "\nQuality mode: perform deep multi-angle research. Use __reasoning_preamble before non-done tools, follow up on gaps, and prefer comprehensive coverage before done."
	default:
		return base + "\nBalanced mode: use __reasoning_preamble before non-done tools and gather enough evidence without over-searching."
	}
}

func researchActionDescriptions(req ResearchRequest) string {
	var parts []string
	if req.Mode != ModeSpeed {
		parts = append(parts, `<tool name="__reasoning_preamble">State a concise natural-language plan before other tools.</tool>`)
	}
	if hasSource(req.Sources, SearchSourceWeb) {
		parts = append(parts, `<tool name="web_search">Search the web for current or factual information.</tool>`)
	}
	if hasSource(req.Sources, SearchSourceAcademic) && req.Classification.AcademicSearch {
		parts = append(parts, `<tool name="academic_search">Search academic, paper, benchmark, study, official report, or research sources.</tool>`)
	}
	if hasSource(req.Sources, SearchSourceDiscussions) && req.Classification.DiscussionSearch {
		parts = append(parts, `<tool name="social_search">Search forums, issues, community feedback, and discussion sources.</tool>`)
	}
	if len(req.FileIDs) > 0 {
		parts = append(parts, `<tool name="uploads_search">Search uploaded files.</tool>`)
	}
	parts = append(parts, `<tool name="scrape_url">Scrape exact URLs when requested or needed for deep reading.</tool>`)
	parts = append(parts, `<tool name="done">Signal research completion.</tool>`)
	return strings.Join(parts, "\n")
}

func collectToolCalls(ctx context.Context, ch <-chan *model.Response) ([]model.ToolCall, error) {
	callsByID := map[string]model.ToolCall{}
	var ordered []string
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case rsp, ok := <-ch:
			if !ok {
				var out []model.ToolCall
				for _, id := range ordered {
					out = append(out, callsByID[id])
				}
				return out, nil
			}
			if rsp == nil || rsp.Error != nil {
				if rsp != nil && rsp.Error != nil {
					return nil, rsp.Error
				}
				continue
			}
			for _, choice := range rsp.Choices {
				for _, call := range append(choice.Message.ToolCalls, choice.Delta.ToolCalls...) {
					key := call.ID
					if key == "" && call.Index != nil {
						key = fmt.Sprintf("index-%d", *call.Index)
					}
					if key == "" {
						key = fmt.Sprintf("call-%d", len(ordered)+1)
					}
					existing, ok := callsByID[key]
					if !ok {
						ordered = append(ordered, key)
						callsByID[key] = call
						continue
					}
					if call.ID != "" {
						existing.ID = call.ID
					}
					if call.Function.Name != "" {
						existing.Function.Name += call.Function.Name
					}
					if len(call.Function.Arguments) > 0 {
						existing.Function.Arguments = append(existing.Function.Arguments, call.Function.Arguments...)
					}
					callsByID[key] = existing
				}
			}
		}
	}
}

func formatMessagesForPrompt(messages []model.Message) string {
	var b strings.Builder
	for _, msg := range messages {
		text := messageText(msg)
		if text == "" {
			continue
		}
		fmt.Fprintf(&b, "%s: %s\n", msg.Role, text)
	}
	return strings.TrimSpace(b.String())
}

func tailMessages(messages []model.Message, n int) []model.Message {
	if len(messages) <= n {
		return messages
	}
	return messages[len(messages)-n:]
}

func extractJSONObject(text string) string {
	raw := strings.TrimSpace(text)
	if start := strings.Index(raw, "{"); start >= 0 {
		raw = raw[start:]
	}
	if end := strings.LastIndex(raw, "}"); end >= 0 {
		raw = raw[:end+1]
	}
	return raw
}

func callArgsMap(raw []byte) map[string]any {
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil
	}
	return out
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func resultTexts(results []SearchResult) []string {
	out := make([]string, 0, len(results))
	for _, result := range results {
		out = append(out, result.Title+"\n"+result.Content)
	}
	return out
}

func dedupeResults(results []SearchResult) []SearchResult {
	seen := map[string]bool{}
	var out []SearchResult
	for _, result := range results {
		key := canonicalResultKey(result)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, result)
	}
	return out
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func splitText(value string, chunkSize, overlap int) []string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) <= chunkSize {
		if value == "" {
			return nil
		}
		return []string{value}
	}
	if overlap >= chunkSize {
		overlap = chunkSize / 4
	}
	var chunks []string
	for start := 0; start < len(value); {
		end := start + chunkSize
		if end > len(value) {
			end = len(value)
		}
		chunks = append(chunks, value[start:end])
		if end == len(value) {
			break
		}
		start = end - overlap
	}
	return chunks
}
