package vane

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
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
	seen := map[string]int{}
	var firstErr error
	usedTools := false
	failedInfoCalls := 0
	informationCalls := 0
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
		type callOutcome struct {
			call         model.ToolCall
			name         string
			args         map[string]any
			actionResult researchActionResult
			err          error
		}
		outcomes := make([]callOutcome, len(toolCalls))
		var wg sync.WaitGroup
		sem := make(chan struct{}, effectiveConcurrency(req.Mode, firstPositive(req.Concurrency, r.Concurrency)))
		for i, call := range toolCalls {
			name := call.Function.Name
			args := callArgsMap(call.Function.Arguments)
			outcomes[i] = callOutcome{call: call, name: name, args: args}
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
			wg.Add(1)
			go func(i int, call model.ToolCall, name string) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				actionResult, err := r.executeResearchTool(ctx, req, name, call.Function.Arguments, step+1)
				outcomes[i].actionResult = actionResult
				outcomes[i].err = err
			}(i, call, name)
		}
		wg.Wait()
		done := false
		for _, outcome := range outcomes {
			call := outcome.call
			name := outcome.name
			actionResult := outcome.actionResult
			err := outcome.err
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
				if key == "" {
					continue
				}
				if existing, ok := seen[key]; ok {
					out[existing] = mergeSearchResult(out[existing], result)
					continue
				}
				seen[key] = len(out)
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
			if isInformationTool(name) && len(actionResult.Results) == 0 && (err != nil || actionResult.Error != "") {
				informationCalls++
				failedInfoCalls++
			} else if len(actionResult.Results) > 0 {
				if isInformationTool(name) {
					informationCalls++
				}
				failedInfoCalls = 0
			}
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
		if shouldStopAfterSoftInformationBudget(req.Mode, out, informationCalls, firstPositive(req.SoftMaxInformationCalls, r.SoftMaxInformationCalls)) {
			break
		}
		if failedInfoCalls >= maxFailedInformationToolCalls(req.Mode) {
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

func isInformationTool(name string) bool {
	switch name {
	case "web_search", "academic_search", "social_search", "uploads_search", "scrape_url":
		return true
	default:
		return false
	}
}

func maxFailedInformationToolCalls(mode Mode) int {
	switch mode {
	case ModeQuality:
		return 4
	default:
		return 3
	}
}

func shouldStopAfterSoftInformationBudget(mode Mode, results []SearchResult, calls int, budget int) bool {
	if budget <= 0 || calls < budget {
		return false
	}
	minResults := 4
	if mode == ModeQuality {
		minResults = 8
	}
	return countCredibleResults(results) >= minResults
}

func countCredibleResults(results []SearchResult) int {
	count := 0
	for _, result := range results {
		if strings.TrimSpace(result.URL) != "" || strings.TrimSpace(result.Content) != "" {
			count++
		}
	}
	return count
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
		tools["web_search"] = r.queryTool(req, "web_search", SearchSourceWeb, "Perform web searches for concise targeted queries.")
	}
	if hasSource(req.Sources, SearchSourceAcademic) && req.Classification.AcademicSearch && !req.Classification.SkipSearch {
		tools["academic_search"] = r.queryTool(req, "academic_search", SearchSourceAcademic, "Search scholarly, paper, benchmark, official report, and research-oriented sources.")
	}
	if hasSource(req.Sources, SearchSourceDiscussions) && req.Classification.DiscussionSearch && !req.Classification.SkipSearch {
		tools["social_search"] = r.queryTool(req, "social_search", SearchSourceDiscussions, "Search community discussions, issues, forums, and social feedback.")
	}
	if len(req.FileIDs) > 0 && r.EmbeddingProvider != nil {
		tools["uploads_search"] = r.queryTool(req, "uploads_search", SearchSourceUploads, "Search the user's uploaded files and personal documents.")
	}
	tools["scrape_url"] = function.NewFunctionTool(
		func(ctx context.Context, args scrapeURLArgs) (researchActionResult, error) {
			return r.scrapeURLs(ctx, req, args.URLs)
		},
		function.WithName("scrape_url"),
		function.WithDescription("Scrape and extract content from specific URLs. Use only when the user explicitly asks about exact URLs. NEVER call this tool yourself just to get extra information."),
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

func (r Researcher) queryTool(req ResearchRequest, name string, source SearchSource, description string) tool.Tool {
	return function.NewFunctionTool(
		func(ctx context.Context, args queryActionArgs) (researchActionResult, error) {
			results, err := r.executeQueries(ctx, req, source, args.Queries)
			res := researchActionResult{Type: "search_results", Results: results}
			if err != nil {
				res.Error = err.Error()
			}
			return res, err
		},
		function.WithName(name),
		function.WithDescription(description),
		function.WithInputSchema(&tool.Schema{Type: "object", Required: []string{"queries"}, Properties: map[string]*tool.Schema{
			"queries": {Type: "array", Description: "Search queries, maximum 3. Use targeted SEO-friendly keywords, not full natural-language sentences.", Items: &tool.Schema{Type: "string"}},
		}}),
	)
}

func (r Researcher) executeResearchTool(ctx context.Context, req ResearchRequest, name string, args []byte, step int) (researchActionResult, error) {
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
	queries = r.repairSearchQueries(ctx, req, source, queries)
	if len(queries) > 3 {
		queries = queries[:3]
	}
	var all []SearchResult
	var firstErr error
	type queryOutcome struct {
		results []SearchResult
		err     error
	}
	outcomes := make([]queryOutcome, len(queries))
	var wg sync.WaitGroup
	sem := make(chan struct{}, effectiveConcurrency(req.Mode, firstPositive(req.Concurrency, r.Concurrency)))
	for i, q := range queries {
		i, q := i, resolveRelativeDateQuery(q, req.Now)
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			results, err := r.runAction(ctx, req, source, q)
			for i := range results {
				if results[i].Source == "" {
					results[i].Source = source
				}
			}
			outcomes[i] = queryOutcome{results: results, err: err}
		}()
	}
	wg.Wait()
	for _, outcome := range outcomes {
		if outcome.err != nil && firstErr == nil {
			firstErr = outcome.err
		}
		all = append(all, outcome.results...)
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

func (r Researcher) repairSearchQueries(ctx context.Context, req ResearchRequest, source SearchSource, queries []string) []string {
	if r.ResearchModel == nil || source == SearchSourceUploads || !needsQueryRepair(queries) {
		return queries
	}
	now := time.Now()
	if !req.Now.IsZero() {
		now = req.Now
	}
	var b strings.Builder
	for i, query := range queries {
		fmt.Fprintf(&b, "%d. %s\n", i+1, resolveRelativeDateQuery(query, req.Now))
	}
	ch, err := r.ResearchModel.GenerateContent(ctx, &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage(`You are Vane's search query repairer. Return compact JSON only: {"queries":["..."]}.
Only rewrite bad search queries that look like full user sentences or verbose conversational text.
Return 1 to 3 search-engine-friendly keyword queries.
Preserve entities, locations, dates, and constraints.
Use short Chinese keyword queries for Chinese local/news questions.
If the input queries are already good keyword queries, return them unchanged.`),
			model.NewUserMessage(fmt.Sprintf(
				"Current date: %s\nMode: %s\nSource: %s\nUser query: %s\nStandalone question: %s\nQueries to repair:\n%s",
				now.Format("2006-01-02"),
				req.Mode,
				source,
				req.Query,
				firstNonEmpty(req.Classification.StandaloneFollowUp, req.Query),
				b.String(),
			)),
		},
		GenerationConfig: model.GenerationConfig{Stream: false, Temperature: floatPtr(0), MaxTokens: intPtr(220)},
	})
	if err != nil {
		return queries
	}
	text, err := collectResponseText(ctx, ch)
	if err != nil {
		return queries
	}
	var parsed struct {
		Queries []string `json:"queries"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(text)), &parsed); err != nil {
		return queries
	}
	repaired := sanitizeRepairedQueries(parsed.Queries)
	if len(repaired) == 0 {
		return queries
	}
	return repaired
}

func needsQueryRepair(queries []string) bool {
	for _, query := range queries {
		if looksLikeVerboseSearchQuery(query) {
			return true
		}
	}
	return false
}

func looksLikeVerboseSearchQuery(query string) bool {
	query = strings.TrimSpace(query)
	runes := []rune(query)
	if len(runes) > 38 {
		return true
	}
	verbosePhrases := []string{"搜索", "看看", "分析", "一下", "哪些", "如何", "为什么", "please", "tell me", "what are", "how does"}
	lower := strings.ToLower(query)
	for _, phrase := range verbosePhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return strings.Count(query, " ") > 8 || strings.Count(query, "，")+strings.Count(query, ",") > 1
}

func sanitizeRepairedQueries(queries []string) []string {
	var out []string
	for _, query := range queries {
		query = strings.TrimSpace(query)
		if query == "" || len([]rune(query)) > 80 || looksLikeVerboseSearchQuery(query) {
			continue
		}
		out = append(out, query)
		if len(out) >= 3 {
			break
		}
	}
	return uniqueStrings(out)
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
	candidates := filterQualityResults(queries, results)
	if len(candidates) == 0 {
		candidates = results
	}
	picked := r.pickQualityResults(ctx, queries, candidates)
	if len(picked) == 0 {
		picked = candidates
	}
	picked = filterQualityResults(queries, picked)
	if len(picked) == 0 {
		picked = candidates
	}
	if len(picked) > 3 {
		picked = picked[:3]
	}
	pickedKeys := map[string]bool{}
	for _, result := range picked {
		if key := canonicalResultKey(result); key != "" {
			pickedKeys[key] = true
		}
	}
	outcomes := make([]SearchResult, len(picked))
	var wg sync.WaitGroup
	sem := make(chan struct{}, effectiveConcurrency(req.Mode, firstPositive(req.Concurrency, r.Concurrency)))
	for i, result := range picked {
		i, result := i, result
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			outcomes[i] = r.deepReadOne(ctx, queries, result)
		}()
	}
	wg.Wait()
	var out []SearchResult
	for _, result := range outcomes {
		if strings.TrimSpace(result.Title) == "" && strings.TrimSpace(result.URL) == "" && strings.TrimSpace(result.Content) == "" {
			continue
		}
		if !qualityResultRelevant(queries, result) {
			continue
		}
		out = append(out, result)
	}
	if len(out) == 0 {
		return candidates
	}
	supplementalLimit := min(maxQualitySupplementalResults, totalResultLimit(req.Mode)-len(out))
	out = append(out, supplementalQualityResults(queries, candidates, pickedKeys, supplementalLimit)...)
	return dedupeResults(out)
}

func (r Researcher) deepReadOne(ctx context.Context, queries []string, result SearchResult) SearchResult {
	if strings.TrimSpace(result.URL) == "" {
		result.Stage = "deep_read"
		return result
	}
	doc, err := r.ScrapeProvider.Scrape(ctx, result.URL)
	if err != nil || strings.TrimSpace(doc.Content) == "" {
		result.Stage = "deep_read"
		return result
	}
	facts := r.extractFacts(ctx, queries, doc.Content)
	if strings.TrimSpace(facts) == "" {
		facts = doc.Content
	}
	return SearchResult{
		Title:   firstNonEmpty(doc.Title, result.Title),
		URL:     firstNonEmpty(doc.URL, result.URL),
		Content: facts,
		Source:  result.Source,
		Stage:   "deep_read",
	}
}

const maxQualitySupplementalResults = 8

func supplementalQualityResults(queries []string, results []SearchResult, exclude map[string]bool, limit int) []SearchResult {
	if limit <= 0 {
		return nil
	}
	if limit > maxQualitySupplementalResults {
		limit = maxQualitySupplementalResults
	}
	var out []SearchResult
	for _, result := range results {
		key := canonicalResultKey(result)
		if key == "" || exclude[key] {
			continue
		}
		if !qualitySupplementalRelevant(queries, result) {
			continue
		}
		result.Stage = "supplemental"
		out = append(out, result)
		if len(out) >= limit {
			break
		}
	}
	return out
}

func qualitySupplementalRelevant(queries []string, result SearchResult) bool {
	if !qualityResultRelevant(queries, result) {
		return false
	}
	terms := queryRelevanceTerms(queries)
	if len(terms) == 0 {
		return true
	}
	haystack := strings.ToLower(result.Title + "\n" + result.Content + "\n" + result.URL)
	matches := 0
	for _, term := range terms {
		if strings.Contains(haystack, term) {
			matches++
		}
	}
	if len(terms) <= 2 {
		return matches > 0
	}
	return matches >= 2
}

func filterQualityResults(queries []string, results []SearchResult) []SearchResult {
	filtered := make([]SearchResult, 0, len(results))
	for _, result := range results {
		if qualityResultRelevant(queries, result) {
			filtered = append(filtered, result)
		}
	}
	return filtered
}

func qualityResultRelevant(queries []string, result SearchResult) bool {
	eventTerms := queryEventTerms(queries)
	if len(eventTerms) == 0 {
		return true
	}
	haystack := resultHaystack(result)
	if !containsAnyTerm(haystack, eventTerms) {
		return false
	}
	if genericSearchResult(result) && countTermMatches(haystack, eventTerms) < 2 {
		return false
	}
	return true
}

func resultHaystack(result SearchResult) string {
	return strings.ToLower(result.Title + "\n" + result.Content + "\n" + result.URL)
}

func containsAnyTerm(haystack string, terms []string) bool {
	for _, term := range terms {
		if strings.Contains(haystack, term) {
			return true
		}
	}
	return false
}

func countTermMatches(haystack string, terms []string) int {
	matches := 0
	for _, term := range terms {
		if strings.Contains(haystack, term) {
			matches++
		}
	}
	return matches
}

func queryEventTerms(queries []string) []string {
	catalog := []string{
		"\u5927\u98ce", "\u5f3a\u98ce", "\u98ce\u66b4", "\u9f99\u5377\u98ce", "\u6c99\u5c18\u66b4",
		"\u5f3a\u5bf9\u6d41", "\u66b4\u96e8", "\u51b0\u96f9", "\u53f0\u98ce", "\u707e\u5bb3",
		"\u4e8b\u6545", "\u4f24\u4ea1", "\u53d7\u4f24", "\u635f\u5931", "\u53d7\u635f",
		"\u5012\u584c", "\u5012\u4f0f", "\u5760\u843d", "\u505c\u7535", "\u4ea4\u901a",
		"\u5217\u8f66", "\u822a\u73ed", "\u665a\u70b9", "\u9884\u8b66", "\u901a\u62a5",
		"wind", "storm", "windstorm", "sandstorm", "tornado", "disaster", "accident",
		"damage", "casualty", "casualties", "injured", "outage", "delay", "warning", "alert",
	}
	haystack := strings.ToLower(strings.Join(queries, "\n"))
	seen := map[string]bool{}
	var terms []string
	for _, term := range catalog {
		if strings.Contains(haystack, term) && !seen[term] {
			seen[term] = true
			terms = append(terms, term)
		}
	}
	return terms
}

func genericSearchResult(result SearchResult) bool {
	haystack := resultHaystack(result)
	genericTerms := []string{
		"baike.", "wikipedia.org", "britannica.com", "mafengwo.cn", "chinahighlights.com",
		"chinadiscovery.com", "holafly.com", "travel guide", "things to do", "places to visit",
		"attractions", "facts", "spelling", "pronunciation",
		"\u767e\u79d1", "\u65c5\u6e38", "\u65c5\u6e38\u653b\u7565", "\u666f\u70b9",
		"\u5fc5\u53bb", "\u82f1\u6587", "\u62fc\u5199", "\u600e\u4e48\u8bfb",
	}
	return containsAnyTerm(haystack, genericTerms)
}

func queryRelevanceTerms(queries []string) []string {
	stop := map[string]bool{
		"the": true, "and": true, "for": true, "with": true, "from": true, "into": true,
		"latest": true, "overview": true, "official": true, "report": true, "reports": true,
		"impact": true, "impacts": true, "timeline": true, "statistics": true,
		"搜索": true, "看看": true, "哪些": true, "分析": true, "影响": true, "事故": true,
		"重大": true, "昨日": true, "昨天": true, "官方": true, "通报": true,
	}
	seen := map[string]bool{}
	var terms []string
	for _, query := range queries {
		for _, raw := range strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
			return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') && !(r >= '\u4e00' && r <= '\u9fff')
		}) {
			term := strings.TrimSpace(raw)
			if term == "" || stop[term] || seen[term] {
				continue
			}
			if len([]rune(term)) < 2 {
				continue
			}
			seen[term] = true
			terms = append(terms, term)
			if len(terms) >= 12 {
				return terms
			}
		}
	}
	return terms
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
	chunks := splitText(content, 4000, 500)
	out := make([]string, len(chunks))
	var wg sync.WaitGroup
	sem := make(chan struct{}, effectiveConcurrency(ModeQuality, r.Concurrency))
	for i, chunk := range chunks {
		i, chunk := i, chunk
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			ch, err := r.ResearchModel.GenerateContent(ctx, &model.Request{
				Messages: []model.Message{
					model.NewSystemMessage("You are Vane's information extractor. Return compact JSON only: {\"extracted_facts\":\"- Fact\"}. Extract only facts relevant to the queries, preserve raw numbers and table values, remove boilerplate."),
					model.NewUserMessage(fmt.Sprintf("<queries>%s</queries>\n<scraped_data>%s</scraped_data>", strings.Join(queries, ", "), chunk)),
				},
				GenerationConfig: model.GenerationConfig{Stream: false, Temperature: floatPtr(0), MaxTokens: intPtr(900)},
			})
			if err != nil {
				return
			}
			text, err := collectResponseText(ctx, ch)
			if err != nil {
				return
			}
			var parsed struct {
				ExtractedFacts string `json:"extracted_facts"`
			}
			if err := json.Unmarshal([]byte(extractJSONObject(text)), &parsed); err == nil && strings.TrimSpace(parsed.ExtractedFacts) != "" {
				out[i] = strings.TrimSpace(parsed.ExtractedFacts)
			}
		}()
	}
	wg.Wait()
	return strings.TrimSpace(strings.Join(nonEmptyStrings(out), "\n"))
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
	results := make([]SearchResult, len(urls))
	var wg sync.WaitGroup
	sem := make(chan struct{}, effectiveConcurrency(req.Mode, firstPositive(req.Concurrency, r.Concurrency)))
	for i, rawURL := range urls {
		i, rawURL := i, rawURL
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			doc, err := r.ScrapeProvider.Scrape(ctx, rawURL)
			if err != nil {
				results[i] = SearchResult{Title: "Error scraping " + rawURL, URL: rawURL, Content: err.Error()}
				return
			}
			content := doc.Content
			if req.Mode == ModeQuality {
				if facts := r.extractFacts(ctx, []string{req.Query}, doc.Content); facts != "" {
					content = facts
				}
			}
			results[i] = SearchResult{Title: firstNonEmpty(doc.Title, rawURL), URL: firstNonEmpty(doc.URL, rawURL), Content: content}
		}()
	}
	wg.Wait()
	return researchActionResult{Type: "search_results", Results: compactResults(results)}, nil
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
		return base + `

<speed_mode>
- Act quickly and efficiently.
- Use one strong information-gathering tool call when current or specific facts are needed, then call done.
- If two or three information attempts fail or return no useful results, stop and call done instead of looping.
- Prefer web_search for current facts, basic factual checks, and stale information. Your built-in knowledge may be outdated.
</speed_mode>`
	case ModeQuality:
		return base + `

<quality_mode>
- You are a deep-research orchestrator. Follow an iterative reason-act loop.
- You MUST call __reasoning_preamble before every tool call in this assistant turn, including done. If you do not call it, the following tool call should be treated as invalid.
- Use up to 10 tool calls total in this turn even though the outer loop may allow more iterations.
- Unless the question is very simple, aim for 4-7 information-gathering calls across several web_search calls.
- Start broad, then narrow: overview, official reports, timeline, impact, statistics, disputes, conflicting accounts, missing angles, and reputable local coverage.
- Each web_search call can contain at most 3 concise queries.
- Do not call done early. Call done only after the gathered sources cover the main claims, important gaps, and uncertainty.
- Never output final answer text directly.
</quality_mode>

<research_strategy>
1. Start with broad overview queries to establish what happened.
2. Follow up with targeted searches for official notices, timelines, casualties or damage, transport or infrastructure impact, weather bureau details, and local reporting.
3. Search for conflicting or updated accounts if the topic is breaking or uncertain.
4. Stop only when the evidence is broad enough for a sourced answer or when repeated attempts fail.
</research_strategy>`
	default:
		return base + `

<balanced_mode>
- You MUST call __reasoning_preamble before every tool call in this assistant turn, including done.
- Use at most 6 tool calls total: reasoning, 2-3 information-gathering calls when needed, reasoning, and done.
- Aim for at least two information calls unless the answer is trivial or the first results are clearly sufficient.
- Do not call done until reasoning plus necessary tool calls are complete.
- Never output final answer text directly.
</balanced_mode>`
	}
}

func researchActionDescriptions(req ResearchRequest) string {
	var parts []string
	if req.Mode != ModeSpeed {
		parts = append(parts, `<tool name="__reasoning_preamble">State a concise natural-language plan before other tools.</tool>`)
	}
	if hasSource(req.Sources, SearchSourceWeb) {
		parts = append(parts, fmt.Sprintf(`<tool name="web_search">%s</tool>`, webSearchActionDescription(req.Mode)))
	}
	if hasSource(req.Sources, SearchSourceAcademic) && req.Classification.AcademicSearch {
		parts = append(parts, `<tool name="academic_search">Use this tool to perform academic searches for scholarly articles, papers, and research studies relevant to the user's query. Provide up to 3 concise search queries. Make sure the queries are specific and relevant to the user's needs.</tool>`)
	}
	if hasSource(req.Sources, SearchSourceDiscussions) && req.Classification.DiscussionSearch {
		parts = append(parts, `<tool name="social_search">Use this tool to perform social media searches for relevant posts, discussions, and trends related to the user's query. Provide up to 3 concise search queries. Make sure the queries are specific and relevant to the user's needs.</tool>`)
	}
	if len(req.FileIDs) > 0 {
		parts = append(parts, `<tool name="uploads_search">Search uploaded files.</tool>`)
	}
	parts = append(parts, `<tool name="scrape_url">Scrape exact URLs only when the user explicitly asks about specific web pages. Never call this yourself to get extra information; quality web_search performs deep reading internally.</tool>`)
	parts = append(parts, `<tool name="done">Signal research completion.</tool>`)
	return strings.Join(parts, "\n")
}

func webSearchActionDescription(mode Mode) string {
	common := `Use this tool to perform web searches based on the provided queries. This is useful when you need to gather information from the web to answer the user's questions. You can provide up to 3 queries at a time. You will have to use this every single time if this is present and relevant.

Your queries should be very targeted and specific to the information you need, avoid broad or generic queries.
Your queries shouldn't be sentences but rather keywords that are SEO friendly and can be used to search the web for information.

You can search for 3 queries in one go, make sure to utilize all 3 queries to maximize the information you can gather. If a question is simple, then split your queries to cover different aspects or related topics to get a comprehensive understanding.`
	switch mode {
	case ModeSpeed:
		return common + `
You are currently on speed mode, meaning you would only get to call this tool once. Make sure to prioritize the most important queries that are likely to get you the needed information in one go.
For example, if the user is asking about the features of a new technology, you might use queries like "GPT-5.1 features", "GPT-5.1 release date", "GPT-5.1 improvements" rather than a broad query like "Tell me about GPT-5.1".
If this tool is present and no other tools are more relevant, you MUST use this tool to get the needed information.`
	case ModeQuality:
		return common + `
You have to call this tool several times to gather enough information unless the question is very simple (like greeting questions or basic facts).
Start initially with broader queries to get an overview, then narrow down with more specific queries based on the results you receive.
Never stop before at least 5-6 iterations of searches unless the user question is very simple.
If this tool is present and no other tools are more relevant, you MUST use this tool to get the needed information. You can call this tool multiple times as needed.`
	default:
		return common + `
You can call this tool several times if needed to gather enough information.
Start initially with broader queries to get an overview, then narrow down with more specific queries based on the results you receive.
For example if the user is asking about Tesla, your actions should be like:
1. __reasoning_preamble "The user is asking about Tesla. I will start with broader queries to get an overview of Tesla, then narrow down with more specific queries based on the results I receive." then
2. web_search ["Tesla", "Tesla latest news", "Tesla stock price"] then
3. __reasoning_preamble "Based on the previous search results, I will now narrow down my queries to focus on Tesla's recent developments and stock performance." then
4. web_search ["Tesla Q2 2025 earnings", "Tesla new model 2025", "Tesla stock analysis"] then done.
5. __reasoning_preamble "I have gathered enough information to provide a comprehensive answer."
6. done.
If this tool is present and no other tools are more relevant, you MUST use this tool to get the needed information. You can call this tool multiple times as needed.`
	}
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

func effectiveConcurrency(mode Mode, configured int) int {
	if configured > 0 {
		return clampInt(configured, 1, 12)
	}
	switch mode {
	case ModeSpeed:
		return 2
	case ModeQuality:
		return 4
	default:
		return 3
	}
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func nonEmptyStrings(values []string) []string {
	out := values[:0]
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func compactResults(results []SearchResult) []SearchResult {
	out := results[:0]
	for _, result := range results {
		if strings.TrimSpace(result.Title) == "" && strings.TrimSpace(result.URL) == "" && strings.TrimSpace(result.Content) == "" {
			continue
		}
		out = append(out, result)
	}
	return out
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
	seen := map[string]int{}
	var out []SearchResult
	for _, result := range results {
		key := canonicalResultKey(result)
		if key == "" {
			continue
		}
		if existing, ok := seen[key]; ok {
			out[existing] = mergeSearchResult(out[existing], result)
			continue
		}
		seen[key] = len(out)
		out = append(out, result)
	}
	return out
}

func mergeSearchResult(existing, next SearchResult) SearchResult {
	if strings.TrimSpace(existing.Title) == "" {
		existing.Title = next.Title
	}
	if strings.TrimSpace(existing.URL) == "" {
		existing.URL = next.URL
	}
	if existing.Source == "" {
		existing.Source = next.Source
	}
	nextContent := strings.TrimSpace(next.Content)
	if nextContent != "" && !strings.Contains(existing.Content, nextContent) {
		if strings.TrimSpace(existing.Content) != "" {
			existing.Content = strings.TrimSpace(existing.Content) + "\n\n" + nextContent
		} else {
			existing.Content = nextContent
		}
	}
	return existing
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
