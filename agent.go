package vane

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

type SearchAgent struct {
	ClassifierModel   model.Model
	SearchProvider    SearchProvider
	EmbeddingProvider EmbeddingProvider
	WidgetProviders   WidgetProviders
	OnSearchEvent     func(context.Context, SearchEvent)
}

type SearchAgentRequest struct {
	Query         string
	Messages      []model.Message
	Mode          Mode
	Sources       []SearchSource
	FileIDs       []string
	MaxIterations int
	Now           time.Time
}

type SearchAgentResult struct {
	Classification Classification
	Sources        []SearchResult
	Widgets        []WidgetResult
}

func (a SearchAgent) Run(ctx context.Context, req SearchAgentRequest) (SearchAgentResult, error) {
	mode := req.Mode
	if mode == "" {
		mode = ModeBalanced
	}
	classification := a.classify(ctx, req)
	emitSearchEvent(ctx, a.OnSearchEvent, SearchEvent{
		Type:           SearchEventClassification,
		Mode:           mode,
		Query:          req.Query,
		Classification: &classification,
	})
	widgets := a.runWidgets(ctx, req.Query, classification)
	if !classification.ShouldSearch {
		if len(widgets) > 0 {
			emitSearchEvent(ctx, a.OnSearchEvent, SearchEvent{
				Type:  SearchEventSourceBlock,
				Mode:  mode,
				Query: req.Query,
				SourceBlock: &SourceBlock{
					Widgets: widgets,
				},
			})
		}
		return SearchAgentResult{Classification: classification, Widgets: widgets}, nil
	}
	researcher := Researcher{
		SearchProvider:    a.SearchProvider,
		EmbeddingProvider: a.EmbeddingProvider,
		OnSearchEvent:     a.OnSearchEvent,
	}
	sources, err := researcher.Research(ctx, ResearchRequest{
		Query:         req.Query,
		Mode:          mode,
		Sources:       classification.Sources,
		FileIDs:       req.FileIDs,
		MaxIterations: req.MaxIterations,
		Now:           req.Now,
	})
	emitSearchEvent(ctx, a.OnSearchEvent, SearchEvent{
		Type:        SearchEventSourceBlock,
		Mode:        mode,
		Query:       req.Query,
		ResultCount: len(sources),
		SourceBlock: &SourceBlock{
			Results: sources,
			Widgets: widgets,
		},
		Results: sources,
	})
	return SearchAgentResult{
		Classification: classification,
		Sources:        sources,
		Widgets:        widgets,
	}, err
}

func (a SearchAgent) classify(ctx context.Context, req SearchAgentRequest) Classification {
	fallback := classifySearch(req)
	if a.ClassifierModel == nil || strings.TrimSpace(req.Query) == "" {
		return fallback
	}
	classifyCtx, cancel := context.WithTimeout(ctx, classifierTimeout(req.Mode))
	defer cancel()
	cfg := model.GenerationConfig{
		Stream:      false,
		Temperature: floatPtr(0),
		MaxTokens:   intPtr(360),
	}
	ch, err := a.ClassifierModel.GenerateContent(classifyCtx, &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage(classifierSystemPrompt()),
			model.NewUserMessage(classifierUserPrompt(req)),
		},
		GenerationConfig: cfg,
	})
	if err != nil {
		return fallback
	}
	text, err := collectResponseText(classifyCtx, ch)
	if err != nil || strings.TrimSpace(text) == "" {
		return fallback
	}
	classification, err := parseClassifierJSON(text, req, fallback)
	if err != nil {
		return fallback
	}
	return classification
}

func classifierTimeout(mode Mode) time.Duration {
	switch mode {
	case ModeSpeed:
		return 2500 * time.Millisecond
	case ModeQuality:
		return 8 * time.Second
	default:
		return 5 * time.Second
	}
}

func classifierSystemPrompt() string {
	return `You are Vane's search intent classifier. Return only compact JSON.

Schema:
{
  "should_search": boolean,
  "intent": string,
  "reason": string,
  "sources": ["web" | "discussions" | "academic"],
  "need_weather": boolean,
  "need_stock": boolean,
  "need_calc": boolean
}

Rules:
- Always include "web" when should_search is true.
- Add "academic" for papers, benchmarks, studies, research evidence, official reports, scientific/medical/policy/technical evaluation questions.
- Add "discussions" for community feedback, real user experience, bugs, issues, complaints, Reddit/HN/forum/GitHub issue style questions.
- Do not include uploads; file search is controlled outside this classifier.
- For casual greetings or pure chat, set should_search=false.
- For arithmetic, set need_calc=true.`
}

func classifierUserPrompt(req SearchAgentRequest) string {
	return fmt.Sprintf("Mode: %s\nEnabled baseline sources: %s\nLatest user query:\n%s", req.Mode, joinSources(normalizeSources(req.Sources, req.FileIDs)), strings.TrimSpace(req.Query))
}

func joinSources(sources []SearchSource) string {
	values := make([]string, 0, len(sources))
	for _, source := range sources {
		values = append(values, string(source))
	}
	return strings.Join(values, ", ")
}

type classifierJSON struct {
	ShouldSearch *bool          `json:"should_search"`
	Intent       string         `json:"intent"`
	Reason       string         `json:"reason"`
	Sources      []SearchSource `json:"sources"`
	NeedWeather  *bool          `json:"need_weather"`
	NeedStock    *bool          `json:"need_stock"`
	NeedCalc     *bool          `json:"need_calc"`
}

func parseClassifierJSON(text string, req SearchAgentRequest, fallback Classification) (Classification, error) {
	raw := strings.TrimSpace(text)
	if start := strings.Index(raw, "{"); start >= 0 {
		raw = raw[start:]
	}
	if end := strings.LastIndex(raw, "}"); end >= 0 {
		raw = raw[:end+1]
	}
	var parsed classifierJSON
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return Classification{}, err
	}
	out := fallback
	if parsed.ShouldSearch != nil {
		out.ShouldSearch = *parsed.ShouldSearch
	}
	if strings.TrimSpace(parsed.Intent) != "" {
		out.Intent = strings.TrimSpace(parsed.Intent)
	}
	if strings.TrimSpace(parsed.Reason) != "" {
		out.Reason = strings.TrimSpace(parsed.Reason)
	}
	if parsed.NeedWeather != nil {
		out.NeedWeather = *parsed.NeedWeather
	}
	if parsed.NeedStock != nil {
		out.NeedStock = *parsed.NeedStock
	}
	if parsed.NeedCalc != nil {
		out.NeedCalc = *parsed.NeedCalc
	}
	if out.ShouldSearch {
		out.Sources = classifierSources(req, parsed.Sources)
		out.SkipReason = ""
	} else if out.SkipReason == "" {
		out.SkipReason = "model classified as no-search"
	}
	out.NeedUploads = len(req.FileIDs) > 0
	if out.NeedUploads && !hasSource(out.Sources, SearchSourceUploads) {
		out.Sources = append(out.Sources, SearchSourceUploads)
	}
	return out, nil
}

func classifierSources(req SearchAgentRequest, modelSources []SearchSource) []SearchSource {
	sources := normalizeSources(append([]SearchSource{}, req.Sources...), req.FileIDs)
	for _, source := range modelSources {
		switch source {
		case SearchSourceWeb, SearchSourceDiscussions, SearchSourceAcademic:
			if !hasSource(sources, source) {
				sources = append(sources, source)
			}
		}
	}
	if !hasSource(sources, SearchSourceWeb) {
		sources = append([]SearchSource{SearchSourceWeb}, sources...)
	}
	return limitAutoSources(req.Mode, sources)
}

func limitAutoSources(mode Mode, sources []SearchSource) []SearchSource {
	if mode != ModeSpeed {
		return sources
	}
	out := []SearchSource{SearchSourceWeb}
	for _, source := range sources {
		if source != SearchSourceWeb {
			out = append(out, source)
			break
		}
	}
	return out
}

func collectResponseText(ctx context.Context, ch <-chan *model.Response) (string, error) {
	var b strings.Builder
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case rsp, ok := <-ch:
			if !ok {
				return b.String(), nil
			}
			b.WriteString(responseText(rsp))
		}
	}
}

func classifySearch(req SearchAgentRequest) Classification {
	query := strings.TrimSpace(req.Query)
	lower := strings.ToLower(query)
	sources := normalizeSources(req.Sources, req.FileIDs)
	classification := Classification{
		ShouldSearch: true,
		Intent:       "web_research",
		Reason:       "The user enabled Vane search, so Vane will gather sources before answering.",
		Sources:      sources,
	}
	if query == "" {
		classification.ShouldSearch = false
		classification.SkipReason = "empty query"
		return classification
	}
	if looksConversational(lower) {
		classification.ShouldSearch = false
		classification.Intent = "conversation"
		classification.SkipReason = "conversational query"
		classification.Reason = "The latest user message does not need external sources."
	}
	classification.NeedWeather = containsAny(lower, "weather", "forecast", "temperature", "rain", "天气", "气温", "下雨", "预报")
	classification.NeedStock = containsAny(lower, "stock", "share price", "ticker", "nasdaq", "nyse", "股票", "股价", "行情")
	classification.NeedCalc = looksLikeCalculation(lower)
	classification.NeedUploads = len(req.FileIDs) > 0
	if classification.NeedWeather {
		classification.Intent = "weather"
	}
	if classification.NeedStock {
		classification.Intent = "stock"
	}
	if classification.NeedCalc {
		classification.Intent = "calculation"
	}
	if classification.NeedUploads && !hasSource(classification.Sources, SearchSourceUploads) {
		classification.Sources = append(classification.Sources, SearchSourceUploads)
	}
	return classification
}

func normalizeSources(sources []SearchSource, fileIDs []string) []SearchSource {
	if len(sources) == 0 {
		sources = []SearchSource{SearchSourceWeb}
	}
	seen := map[SearchSource]bool{}
	out := make([]SearchSource, 0, len(sources)+1)
	for _, source := range sources {
		switch source {
		case SearchSourceWeb, SearchSourceDiscussions, SearchSourceAcademic, SearchSourceUploads:
		default:
			source = SearchSourceWeb
		}
		if !seen[source] {
			seen[source] = true
			out = append(out, source)
		}
	}
	if len(fileIDs) > 0 && !seen[SearchSourceUploads] {
		out = append(out, SearchSourceUploads)
	}
	return out
}

func (a SearchAgent) runWidgets(ctx context.Context, query string, classification Classification) []WidgetResult {
	var widgets []WidgetResult
	run := func(kind WidgetKind, provider WidgetProvider) {
		result := WidgetResult{Kind: kind, Query: query}
		if provider == nil {
			result.Error = "widget provider is not configured"
		} else {
			got, err := provider.RunWidget(ctx, query)
			if err != nil {
				result.Error = err.Error()
			} else {
				result = got
				if result.Kind == "" {
					result.Kind = kind
				}
				if result.Query == "" {
					result.Query = query
				}
			}
		}
		widgets = append(widgets, result)
		emitSearchEvent(ctx, a.OnSearchEvent, SearchEvent{
			Type:   SearchEventWidget,
			Query:  query,
			Widget: &result,
		})
	}
	if classification.NeedWeather {
		run(WidgetWeather, a.WidgetProviders.Weather)
	}
	if classification.NeedStock {
		run(WidgetStock, a.WidgetProviders.Stock)
	}
	if classification.NeedCalc {
		run(WidgetCalc, firstWidgetProvider(a.WidgetProviders.Calculation, calculationWidgetProvider{}))
	}
	return widgets
}

type Researcher struct {
	SearchProvider    SearchProvider
	EmbeddingProvider EmbeddingProvider
	OnSearchEvent     func(context.Context, SearchEvent)
}

type ResearchRequest struct {
	Query         string
	Mode          Mode
	Sources       []SearchSource
	FileIDs       []string
	MaxIterations int
	Now           time.Time
}

func (r Researcher) Research(ctx context.Context, req ResearchRequest) ([]SearchResult, error) {
	iterations := req.MaxIterations
	if iterations <= 0 {
		iterations = defaultMaxIterations(req.Mode)
	}
	queries := buildResearchQueries(req.Query, req.Mode, iterations, req.Now)
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:       SearchEventStart,
		Mode:       req.Mode,
		Query:      req.Query,
		Queries:    append([]string(nil), queries...),
		QueryTotal: len(queries),
	})
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:      SearchEventResearchStart,
		Mode:      req.Mode,
		Query:     req.Query,
		StepTotal: iterations,
		Queries:   append([]string(nil), queries...),
	})
	seen := map[string]bool{}
	var out []SearchResult
	var firstErr error
	for step, query := range queries {
		if len(out) >= totalResultLimit(req.Mode) {
			break
		}
		emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
			Type:       SearchEventQuery,
			Mode:       req.Mode,
			Query:      query,
			QueryIndex: step + 1,
			QueryTotal: len(queries),
		})
		emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
			Type:      SearchEventResearchStep,
			Mode:      req.Mode,
			Query:     query,
			Step:      step + 1,
			StepTotal: iterations,
			Message:   fmt.Sprintf("Research step %d/%d", step+1, iterations),
		})
		for _, source := range req.Sources {
			if len(out) >= totalResultLimit(req.Mode) {
				break
			}
			results, err := r.executeSourceAction(ctx, req, source, query, step+1)
			if err != nil && firstErr == nil {
				firstErr = err
			}
			before := len(out)
			for _, result := range results {
				if result.Source == "" {
					result.Source = source
				}
				key := canonicalResultKey(result)
				if key == "" || seen[key] {
					continue
				}
				seen[key] = true
				out = append(out, result)
			}
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
	}
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:        SearchEventEnd,
		Mode:        req.Mode,
		Query:       req.Query,
		ResultCount: len(out),
		Results:     append([]SearchResult(nil), out...),
	})
	return out, firstErr
}

func (r Researcher) executeSourceAction(ctx context.Context, req ResearchRequest, source SearchSource, query string, step int) ([]SearchResult, error) {
	actionName := actionNameForSource(source)
	callID := fmt.Sprintf("%s-%d", actionName, step)
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:   SearchEventToolCall,
		Mode:   req.Mode,
		Query:  query,
		Action: actionName,
		Step:   step,
		ToolCall: &ToolCallEvent{
			ID:     callID,
			Name:   actionName,
			Query:  query,
			Step:   step,
			Source: source,
		},
	})
	results, err := r.runAction(ctx, req, source, query)
	if err == nil {
		for i := range results {
			if results[i].Source == "" {
				results[i].Source = source
			}
		}
	}
	resultEvent := &ToolResultEvent{
		ID:          callID,
		Name:        actionName,
		Query:       query,
		ResultCount: len(results),
		Results:     results,
		Step:        step,
		Source:      source,
	}
	if err != nil {
		resultEvent.Error = err.Error()
	}
	emitSearchEvent(ctx, r.OnSearchEvent, SearchEvent{
		Type:        SearchEventToolResult,
		Mode:        req.Mode,
		Query:       query,
		Action:      actionName,
		Step:        step,
		ToolResult:  resultEvent,
		ResultCount: len(results),
		Results:     results,
	})
	return results, err
}

func (r Researcher) runAction(ctx context.Context, req ResearchRequest, source SearchSource, query string) ([]SearchResult, error) {
	switch source {
	case SearchSourceUploads:
		if r.EmbeddingProvider == nil || len(req.FileIDs) == 0 {
			return nil, nil
		}
		return r.EmbeddingProvider.SearchUploads(ctx, query, req.FileIDs, resultsPerQuery(req.Mode))
	default:
		if r.SearchProvider == nil {
			return nil, fmt.Errorf("vane: search provider is nil")
		}
		return r.SearchProvider.Search(ctx, query, SearchOptions{Page: 1, Limit: resultsPerQuery(req.Mode), Source: source})
	}
}

func buildResearchQueries(query string, mode Mode, iterations int, now time.Time) []string {
	expansion := expandTemporalQuery(query, now)
	query = expansion.Query
	base := buildQueries(query, mode)
	base = append(base, expansion.Variants...)
	extras := []string{
		query + " key facts",
		query + " recent developments",
		query + " expert analysis",
		query + " opposing views",
		query + " primary sources",
		query + " statistics",
		query + " timeline",
		query + " implications",
		query + " official source",
		query + " limitations",
		query + " background",
		query + " definition",
		query + " examples",
		query + " comparison",
		query + " case study",
		query + " methodology",
		query + " data",
		query + " report",
		query + " policy",
		query + " market",
		query + " technical details",
		query + " risks",
		query + " benefits",
		query + " FAQ",
		query + " 2026",
	}
	queries := uniqueStrings(append(base, extras...))
	if len(queries) > iterations {
		return queries[:iterations]
	}
	return queries
}

type temporalQueryExpansion struct {
	Query    string
	Variants []string
	Dates    []string
}

func expandTemporalQuery(query string, now time.Time) temporalQueryExpansion {
	resolved := resolveRelativeDateQuery(query, now)
	expansion := temporalQueryExpansion{Query: resolved}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	for _, date := range relativeDateMatches(query, now) {
		expansion.Dates = append(expansion.Dates, date.Label)
	}
	if len(expansion.Dates) == 0 {
		return expansion
	}
	terms := eventImpactTerms(query)
	for _, date := range expansion.Dates {
		expansion.Variants = append(expansion.Variants, resolved+" "+date)
		for _, term := range terms {
			expansion.Variants = append(expansion.Variants, resolved+" "+date+" "+term)
		}
	}
	return expansion
}

type relativeDateMatch struct {
	Label      string
	ISO        string
	RangeLabel string
	Phrases    []string
}

func resolveRelativeDateQuery(query string, now time.Time) string {
	if strings.TrimSpace(query) == "" {
		return query
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	out := query
	for _, replacement := range relativeDateReplacements(query, now) {
		out = strings.ReplaceAll(out, replacement.from, replacement.to)
		out = strings.ReplaceAll(out, strings.ToUpper(replacement.from), replacement.to)
		out = strings.ReplaceAll(out, titleASCII(replacement.from), replacement.to)
	}
	return out
}

func relativeDateReplacements(query string, now time.Time) []struct{ from, to string } {
	matches := relativeDateMatches(query, now)
	out := make([]struct{ from, to string }, 0, len(matches)*4)
	for _, match := range matches {
		to := match.Label
		if match.RangeLabel != "" {
			to = match.RangeLabel
		}
		for _, phrase := range match.Phrases {
			if isASCII(phrase) {
				to = match.ISO
				if match.RangeLabel != "" {
					to = match.RangeLabel
				}
			}
			out = append(out, struct{ from, to string }{from: phrase, to: to})
		}
	}
	return out
}

func relativeDateMatches(query string, now time.Time) []relativeDateMatch {
	lower := strings.ToLower(query)
	addDay := func(days int, phrases ...string) relativeDateMatch {
		date := now.AddDate(0, 0, days)
		return relativeDateMatch{
			Label:   fmt.Sprintf("%d\u5e74%d\u6708%d\u65e5", date.Year(), int(date.Month()), date.Day()),
			ISO:     date.Format("2006-01-02"),
			Phrases: phrases,
		}
	}
	var matches []relativeDateMatch
	if phrases := matchedRelativePhrases(query, lower, "\u4eca\u5929", "\u4eca\u65e5", "today"); len(phrases) > 0 {
		matches = append(matches, addDay(0, phrases...))
	}
	if phrases := matchedRelativePhrases(query, lower, "\u6628\u5929", "\u6628\u65e5", "yesterday"); len(phrases) > 0 {
		matches = append(matches, addDay(-1, phrases...))
	}
	if phrases := matchedRelativePhrases(query, lower, "\u524d\u5929", "day before yesterday"); len(phrases) > 0 {
		matches = append(matches, addDay(-2, phrases...))
	}
	if phrases := matchedRelativePhrases(query, lower, "\u4e0a\u5468", "last week"); len(phrases) > 0 {
		start := now.AddDate(0, 0, -7)
		matches = append(matches, relativeDateMatch{
			Label:      fmt.Sprintf("%d\u5e74%d\u6708%d\u65e5", start.Year(), int(start.Month()), start.Day()),
			ISO:        start.Format("2006-01-02"),
			RangeLabel: fmt.Sprintf("%s..%s", start.Format("2006-01-02"), now.Format("2006-01-02")),
			Phrases:    phrases,
		})
	}
	if phrases := matchedRelativePhrases(query, lower, "\u6700\u8fd1", "\u8fd1\u65e5", "recent", "latest"); len(phrases) > 0 {
		start := now.AddDate(0, 0, -7)
		matches = append(matches, relativeDateMatch{
			Label:      fmt.Sprintf("%d\u5e74%d\u6708%d\u65e5", start.Year(), int(start.Month()), start.Day()),
			ISO:        start.Format("2006-01-02"),
			RangeLabel: fmt.Sprintf("%s..%s", start.Format("2006-01-02"), now.Format("2006-01-02")),
			Phrases:    phrases,
		})
	}
	return matches
}

func matchedRelativePhrases(query, lower string, phrases ...string) []string {
	var out []string
	for _, phrase := range phrases {
		if isASCII(phrase) {
			if strings.Contains(lower, strings.ToLower(phrase)) {
				out = append(out, phrase)
			}
			continue
		}
		if strings.Contains(query, phrase) {
			out = append(out, phrase)
		}
	}
	return out
}

func isASCII(value string) bool {
	for i := 0; i < len(value); i++ {
		if value[i] > 127 {
			return false
		}
	}
	return true
}

func eventImpactTerms(query string) []string {
	lower := strings.ToLower(query)
	if !containsAny(query, "\u5927\u98ce", "\u9f99\u5377\u98ce", "\u66b4\u96e8", "\u707e\u5bb3", "\u4e8b\u6545", "\u5f71\u54cd", "\u4f24\u4ea1", "\u635f\u5931") &&
		!containsAny(lower, "wind", "tornado", "storm", "disaster", "accident", "impact", "damage", "casualty") {
		return nil
	}
	return []string{
		"\u4e8b\u6545",
		"\u4f24\u4ea1",
		"\u635f\u5931",
		"\u4ea4\u901a",
		"\u505c\u7535",
		"\u505c\u8fd0",
		"\u5b98\u65b9\u901a\u62a5",
		"impact",
		"damage",
		"casualties",
		"official report",
	}
}

func titleASCII(value string) string {
	if value == "" {
		return ""
	}
	return strings.ToUpper(value[:1]) + value[1:]
}

func defaultMaxIterations(mode Mode) int {
	switch mode {
	case ModeSpeed:
		return 2
	case ModeQuality:
		return 25
	default:
		return 6
	}
}

func actionNameForSource(source SearchSource) string {
	switch source {
	case SearchSourceAcademic:
		return "academic_search"
	case SearchSourceDiscussions:
		return "social_search"
	case SearchSourceUploads:
		return "uploads_search"
	default:
		return "web_search"
	}
}

type calculationWidgetProvider struct{}

func (calculationWidgetProvider) RunWidget(_ context.Context, query string) (WidgetResult, error) {
	value, err := evaluateSimpleExpression(query)
	if err != nil {
		return WidgetResult{Kind: WidgetCalc, Query: query, Error: err.Error()}, nil
	}
	return WidgetResult{
		Kind:    WidgetCalc,
		Query:   query,
		Title:   "Calculation",
		Content: strconv.FormatFloat(value, 'f', -1, 64),
		Data: map[string]any{
			"value": value,
		},
	}, nil
}

func firstWidgetProvider(primary WidgetProvider, fallback WidgetProvider) WidgetProvider {
	if primary != nil {
		return primary
	}
	return fallback
}

func evaluateSimpleExpression(query string) (float64, error) {
	re := regexp.MustCompile(`[-+*/().0-9\s]+`)
	expr := strings.TrimSpace(re.FindString(query))
	if expr == "" {
		return 0, fmt.Errorf("no arithmetic expression found")
	}
	p := expressionParser{s: expr}
	value, err := p.parseExpression()
	if err != nil {
		return 0, err
	}
	p.skip()
	if p.pos != len(p.s) {
		return 0, fmt.Errorf("unsupported expression")
	}
	return value, nil
}

type expressionParser struct {
	s   string
	pos int
}

func (p *expressionParser) parseExpression() (float64, error) {
	left, err := p.parseTerm()
	if err != nil {
		return 0, err
	}
	for {
		p.skip()
		if p.consume('+') {
			right, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left += right
			continue
		}
		if p.consume('-') {
			right, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left -= right
			continue
		}
		return left, nil
	}
}

func (p *expressionParser) parseTerm() (float64, error) {
	left, err := p.parseFactor()
	if err != nil {
		return 0, err
	}
	for {
		p.skip()
		if p.consume('*') {
			right, err := p.parseFactor()
			if err != nil {
				return 0, err
			}
			left *= right
			continue
		}
		if p.consume('/') {
			right, err := p.parseFactor()
			if err != nil {
				return 0, err
			}
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left /= right
			continue
		}
		return left, nil
	}
}

func (p *expressionParser) parseFactor() (float64, error) {
	p.skip()
	if p.consume('(') {
		value, err := p.parseExpression()
		if err != nil {
			return 0, err
		}
		if !p.consume(')') {
			return 0, fmt.Errorf("missing closing parenthesis")
		}
		return value, nil
	}
	start := p.pos
	if p.peek() == '+' || p.peek() == '-' {
		p.pos++
	}
	for p.pos < len(p.s) && ((p.s[p.pos] >= '0' && p.s[p.pos] <= '9') || p.s[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		return 0, fmt.Errorf("expected number")
	}
	return strconv.ParseFloat(strings.TrimSpace(p.s[start:p.pos]), 64)
}

func (p *expressionParser) skip() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t' || p.s[p.pos] == '\n') {
		p.pos++
	}
}

func (p *expressionParser) consume(ch byte) bool {
	p.skip()
	if p.pos < len(p.s) && p.s[p.pos] == ch {
		p.pos++
		return true
	}
	return false
}

func (p *expressionParser) peek() byte {
	if p.pos >= len(p.s) {
		return 0
	}
	return p.s[p.pos]
}

func looksConversational(lower string) bool {
	trimmed := strings.TrimSpace(lower)
	return trimmed == "hi" || trimmed == "hello" || trimmed == "hey" || trimmed == "你好" || trimmed == "谢谢" || trimmed == "thanks"
}

func looksLikeCalculation(lower string) bool {
	if containsAny(lower, "calculate", "compute", "等于多少", "计算", "算一下") {
		return true
	}
	hasDigit := false
	hasOperator := false
	for _, r := range lower {
		if r >= '0' && r <= '9' {
			hasDigit = true
		}
		if strings.ContainsRune("+-*/", r) {
			hasOperator = true
		}
	}
	return hasDigit && hasOperator
}

func containsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func hasSource(sources []SearchSource, source SearchSource) bool {
	for _, candidate := range sources {
		if candidate == source {
			return true
		}
	}
	return false
}
