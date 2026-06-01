package vane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

func TestClassifierSkipsConversationalQuery(t *testing.T) {
	got := classifySearch(SearchAgentRequest{
		Query:   "\u4f60\u597d",
		Mode:    ModeBalanced,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if got.ShouldSearch {
		t.Fatalf("ShouldSearch = true, want false")
	}
	if got.Intent != "conversation" || got.SkipReason == "" {
		t.Fatalf("classification = %#v, want conversational skip", got)
	}
}

func TestClassifierDetectsWidgets(t *testing.T) {
	tests := []struct {
		name string
		text string
		want func(Classification) bool
	}{
		{name: "weather", text: "\u4e0a\u6d77\u4eca\u5929\u7684\u5929\u6c14\u600e\u4e48\u6837", want: func(c Classification) bool { return c.NeedWeather }},
		{name: "stock", text: "NVDA stock price latest", want: func(c Classification) bool { return c.NeedStock }},
		{name: "calc", text: "calculate 12*(3+4)", want: func(c Classification) bool { return c.NeedCalc }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifySearch(SearchAgentRequest{Query: tt.text})
			if !tt.want(got) {
				t.Fatalf("classification = %#v", got)
			}
		})
	}
}

func TestResearcherUsesModeIterationCounts(t *testing.T) {
	tests := []struct {
		mode Mode
		want int
	}{
		{ModeSpeed, 2},
		{ModeBalanced, 6},
		{ModeQuality, 25},
	}
	for _, tt := range tests {
		t.Run(string(tt.mode), func(t *testing.T) {
			queries := buildResearchQueries("vane", tt.mode, defaultMaxIterations(tt.mode), time.Time{})
			if len(queries) != tt.want {
				t.Fatalf("query count = %d, want %d: %#v", len(queries), tt.want, queries)
			}
		})
	}
}

func TestTemporalQueryExpanderResolvesRelativeDates(t *testing.T) {
	now := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	expansion := expandTemporalQuery("search yesterday storm impact", now)
	if strings.Contains(strings.ToLower(expansion.Query), "yesterday") {
		t.Fatalf("query still contains relative date: %q", expansion.Query)
	}
	if !strings.Contains(expansion.Query, "2026-05-31") {
		t.Fatalf("query = %q, want absolute date 2026-05-31", expansion.Query)
	}
	if len(expansion.Dates) == 0 || expansion.Dates[0] != "2026\u5e745\u670831\u65e5" {
		t.Fatalf("dates = %#v, want Chinese date label", expansion.Dates)
	}
}

func TestTemporalQueryExpanderAddsEventImpactVariants(t *testing.T) {
	now := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	query := "\u641c\u7d22\u54c8\u5c14\u6ee8\u6628\u65e5\u7684\u5927\u98ce\uff0c\u770b\u770b\u51fa\u73b0\u4e86\u54ea\u4e9b\u91cd\u5927\u4e8b\u6545\uff0c\u5206\u6790\u4e00\u4e0b\u5e26\u6765\u4e86\u54ea\u4e9b\u5f71\u54cd"
	queries := buildResearchQueries(query, ModeQuality, 25, now)
	if len(queries) == 0 {
		t.Fatal("expected research queries")
	}
	first := queries[0]
	if strings.Contains(first, "\u6628\u65e5") || strings.Contains(first, "\u6628\u5929") {
		t.Fatalf("query still contains relative date: %q", first)
	}
	if !strings.Contains(first, "2026\u5e745\u670831\u65e5") {
		t.Fatalf("query = %q, want absolute date 2026\u5e745\u670831\u65e5", first)
	}
	if !strings.Contains(first, "\u54c8\u5c14\u6ee8") || !strings.Contains(first, "\u5927\u98ce") {
		t.Fatalf("query = %q, want Harbin wind terms", first)
	}
	joined := strings.Join(queries, "\n")
	if !strings.Contains(joined, "\u5b98\u65b9\u901a\u62a5") || !strings.Contains(joined, "\u4f24\u4ea1") {
		t.Fatalf("queries missing disaster/impact variants: %#v", queries)
	}
}

func TestTemporalReplacementOnlyReplacesMatchedPhrase(t *testing.T) {
	now := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	got := resolveRelativeDateQuery("today and yesterday", now)
	if !strings.Contains(got, "2026-06-01") || !strings.Contains(got, "2026-05-31") {
		t.Fatalf("query = %q, want both today and yesterday dates", got)
	}
}

func TestSearchAgentHarbinQuestionUsesAbsoluteDateQueries(t *testing.T) {
	now := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	searcher := &recordingSearchProvider{}
	agent := SearchAgent{SearchProvider: searcher}
	_, err := agent.Run(context.Background(), SearchAgentRequest{
		Query:         "\u641c\u7d22\u54c8\u5c14\u6ee8\u6628\u65e5\u7684\u5927\u98ce\uff0c\u770b\u770b\u51fa\u73b0\u4e86\u54ea\u4e9b\u91cd\u5927\u4e8b\u6545\uff0c\u5206\u6790\u4e00\u4e0b\u5e26\u6765\u4e86\u54ea\u4e9b\u5f71\u54cd",
		Mode:          ModeQuality,
		Sources:       []SearchSource{SearchSourceWeb},
		MaxIterations: 2,
		Now:           now,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(searcher.queries) == 0 {
		t.Fatal("expected search queries")
	}
	for _, query := range searcher.queries {
		if strings.Contains(query, "\u6628\u65e5") || strings.Contains(query, "\u6628\u5929") {
			t.Fatalf("search query still contains relative date: %q", query)
		}
	}
	if !strings.Contains(searcher.queries[0], "2026\u5e745\u670831\u65e5") {
		t.Fatalf("first query = %q, want absolute date", searcher.queries[0])
	}
}

func TestSearchAgentEmitsWidgetAndFallsBackWhenProviderMissing(t *testing.T) {
	var events []SearchEvent
	agent := SearchAgent{
		SearchProvider: &fakeSearchProvider{},
		OnSearchEvent: func(_ context.Context, ev SearchEvent) {
			events = append(events, ev)
		},
	}
	_, err := agent.Run(context.Background(), SearchAgentRequest{
		Query:    "\u4e0a\u6d77\u5929\u6c14",
		Messages: []model.Message{model.NewUserMessage("\u4e0a\u6d77\u5929\u6c14")},
		Mode:     ModeSpeed,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	widget := lastEventOfType(events, SearchEventWidget)
	if widget == nil || widget.Widget == nil || widget.Widget.Kind != WidgetWeather || widget.Widget.Error == "" {
		t.Fatalf("weather widget event = %#v", widget)
	}
}

func TestCalculationWidgetEvaluatesExpression(t *testing.T) {
	got, err := calculationWidgetProvider{}.RunWidget(context.Background(), "calculate 12*(3+4)")
	if err != nil {
		t.Fatalf("RunWidget: %v", err)
	}
	if got.Content != "84" {
		t.Fatalf("content = %q, want 84", got.Content)
	}
}

func TestLLMClassifierAddsAcademicAndDiscussionSources(t *testing.T) {
	classifier := &staticTextModel{text: `{
		"should_search": true,
		"intent": "framework_comparison",
		"reason": "Needs benchmark evidence and community feedback.",
		"sources": ["web", "academic", "discussions"],
		"need_weather": false,
		"need_stock": false,
		"need_calc": true
	}`}
	agent := SearchAgent{
		ClassifierModel: classifier,
		SearchProvider:  &fakeSearchProvider{},
	}
	got := agent.classify(context.Background(), SearchAgentRequest{
		Query:   "Compare LangGraph AutoGen CrewAI benchmarks and community feedback",
		Mode:    ModeBalanced,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if !hasSource(got.Sources, SearchSourceAcademic) || !hasSource(got.Sources, SearchSourceDiscussions) {
		t.Fatalf("sources = %#v, want academic and discussions", got.Sources)
	}
	if !got.NeedCalc {
		t.Fatalf("NeedCalc = false, want true from classifier")
	}
}

func TestVaneStyleClassifierOutputAddsSourcesAndStandaloneFollowUp(t *testing.T) {
	classifier := &staticTextModel{text: `{
		"classification": {
			"skipSearch": false,
			"personalSearch": false,
			"academicSearch": true,
			"discussionSearch": true,
			"showWeatherWidget": false,
			"showStockWidget": false,
			"showCalculationWidget": false
		},
		"standaloneFollowUp": "Compare agent framework benchmarks and community feedback"
	}`}
	agent := SearchAgent{ClassifierModel: classifier}
	got := agent.classify(context.Background(), SearchAgentRequest{
		Query: "那它们 benchmark 和用户反馈呢",
		Messages: []model.Message{
			model.NewUserMessage("LangGraph AutoGen CrewAI 对比"),
			model.NewAssistantMessage("可以从架构和生态比较。"),
			model.NewUserMessage("那它们 benchmark 和用户反馈呢"),
		},
		Mode:    ModeBalanced,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if got.SkipSearch || !got.ShouldSearch {
		t.Fatalf("classification search flags = %#v, want search", got)
	}
	if got.StandaloneFollowUp != "Compare agent framework benchmarks and community feedback" {
		t.Fatalf("StandaloneFollowUp = %q", got.StandaloneFollowUp)
	}
	if !hasSource(got.Sources, SearchSourceAcademic) || !hasSource(got.Sources, SearchSourceDiscussions) {
		t.Fatalf("sources = %#v, want academic and discussions", got.Sources)
	}
}

func TestLLMClassifierFallsBackOnInvalidJSON(t *testing.T) {
	agent := SearchAgent{ClassifierModel: &staticTextModel{text: "not json"}}
	got := agent.classify(context.Background(), SearchAgentRequest{
		Query: "hello",
		Mode:  ModeBalanced,
	})
	if got.ShouldSearch {
		t.Fatalf("ShouldSearch = true, want heuristic conversational fallback")
	}
}

func TestResearcherUsesNativeToolCallLoop(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{
			toolCall("plan-1", "__reasoning_preamble", map[string]any{"plan": "Looking into the current facts."}),
			toolCall("search-1", "web_search", map[string]any{"queries": []string{"vane search agent"}}),
		},
		{
			toolCall("done-1", "done", map[string]any{}),
		},
	}}
	var events []SearchEvent
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
		OnSearchEvent: func(_ context.Context, ev SearchEvent) {
			events = append(events, ev)
		},
	}
	results, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "what is vane",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "what is vane",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeBalanced,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(results) == 0 || len(searcher.queries) != 1 || searcher.queries[0] != "vane search agent" {
		t.Fatalf("results=%#v queries=%#v, want one model-selected search", results, searcher.queries)
	}
	if !hasEventType(events, SearchEventResearchStep) || !hasEventType(events, SearchEventToolCall) || !hasEventType(events, SearchEventToolResult) {
		t.Fatalf("events missing tool-loop events: %#v", events)
	}
	if len(researchModel.requests) < 2 || len(researchModel.requests[0].Tools) == 0 {
		t.Fatalf("research model did not receive tools: %#v", researchModel.requests)
	}
}

func TestQualityResearchScrapesAndExtractsFacts(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"Harbin tornado official report"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
		ScrapeProvider: staticScrapeProvider{content: "Long article: tornado caused power outage and traffic disruption."},
		OnSearchEvent:  func(context.Context, SearchEvent) {},
	}
	results, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "Harbin tornado impacts",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "Harbin tornado impacts",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(results) == 0 || !strings.Contains(results[0].Content, "Extracted tornado fact") {
		t.Fatalf("quality results = %#v, want extracted facts", results)
	}
}

func TestQualityResearchKeepsSupplementalSearchSources(t *testing.T) {
	searcher := &manySearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"Harbin tornado official report"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
		ScrapeProvider: staticScrapeProvider{content: "Long article: tornado caused power outage and traffic disruption."},
	}
	results, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "Harbin tornado impacts",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "Harbin tornado impacts",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(results) <= 3 {
		t.Fatalf("quality results = %d, want deep-read results plus supplemental search sources: %#v", len(results), results)
	}
	if !strings.Contains(results[0].Content, "Extracted tornado fact") {
		t.Fatalf("first result content = %q, want extracted deep-read facts", results[0].Content)
	}
	if !strings.Contains(results[len(results)-1].Content, "Search snippet") {
		t.Fatalf("last result content = %q, want supplemental search snippet", results[len(results)-1].Content)
	}
}

func TestQualitySupplementalSourcesAreFilteredAndCapped(t *testing.T) {
	results := []SearchResult{
		{Title: "Picked", URL: "https://example.com/picked", Content: "Harbin tornado picked"},
	}
	exclude := map[string]bool{"https://example.com/picked": true}
	for i := 0; i < 20; i++ {
		title := fmt.Sprintf("Irrelevant %d", i)
		content := "unrelated entertainment result"
		if i%2 == 0 {
			title = fmt.Sprintf("Harbin tornado local report %d", i)
			content = "Harbin tornado damage and outage details"
		}
		results = append(results, SearchResult{
			Title:   title,
			URL:     fmt.Sprintf("https://example.com/%d", i),
			Content: content,
		})
	}
	got := supplementalQualityResults([]string{"Harbin tornado damage"}, results, exclude, 20)
	if len(got) != maxQualitySupplementalResults {
		t.Fatalf("supplemental len = %d, want cap %d: %#v", len(got), maxQualitySupplementalResults, got)
	}
	for _, result := range got {
		if result.Stage != "supplemental" {
			t.Fatalf("supplemental stage = %q, want supplemental", result.Stage)
		}
		if !strings.Contains(strings.ToLower(result.Title+result.Content), "harbin") {
			t.Fatalf("irrelevant supplemental result kept: %#v", result)
		}
	}
}

func TestQualityResearcherPromptRequiresMultiRoundSearch(t *testing.T) {
	prompt := getResearcherPrompt(ResearchRequest{
		Query: "Harbin tornado impacts",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "Harbin tornado impacts",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	}, 0, defaultMaxIterations(ModeQuality))
	if !strings.Contains(prompt, "aim for 4-7 information-gathering calls") ||
		!strings.Contains(prompt, "You MUST call __reasoning_preamble before every tool call") ||
		!strings.Contains(prompt, "Start broad, then narrow") {
		t.Fatalf("quality researcher prompt missing multi-round guidance: %s", prompt)
	}
}

func TestBalancedResearcherPromptRequiresReasoningPreamble(t *testing.T) {
	prompt := getResearcherPrompt(ResearchRequest{
		Query: "recent model comparison",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "recent model comparison",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeBalanced,
		Sources: []SearchSource{SearchSourceWeb},
	}, 0, defaultMaxIterations(ModeBalanced))
	if !strings.Contains(prompt, "You MUST call __reasoning_preamble before every tool call") ||
		!strings.Contains(prompt, "Use at most 6 tool calls total") {
		t.Fatalf("balanced researcher prompt missing original-style tool guidance: %s", prompt)
	}
}

func TestScrapeURLPromptRequiresExplicitUserURL(t *testing.T) {
	prompt := researchActionDescriptions(ResearchRequest{
		Query: "Harbin tornado impacts",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "Harbin tornado impacts",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if !strings.Contains(prompt, "only when the user explicitly asks about specific web pages") ||
		strings.Contains(prompt, "needed for deep reading") {
		t.Fatalf("scrape_url description should match original explicit-URL constraint: %s", prompt)
	}
}

func TestDedupeResultsMergesDuplicateURLContent(t *testing.T) {
	results := dedupeResults([]SearchResult{
		{Title: "Report", URL: "https://example.com/news#top", Content: "First snippet.", Source: SearchSourceWeb},
		{Title: "Report duplicate", URL: "https://example.com/news", Content: "Second snippet.", Source: SearchSourceWeb},
	})
	if len(results) != 1 {
		t.Fatalf("dedupeResults length = %d, want 1: %#v", len(results), results)
	}
	if !strings.Contains(results[0].Content, "First snippet.") || !strings.Contains(results[0].Content, "Second snippet.") {
		t.Fatalf("dedupeResults did not merge duplicate URL content: %#v", results[0])
	}
}

func TestResearcherToolLoopMergesDuplicateURLContent(t *testing.T) {
	searcher := &duplicateURLSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"first angle"}})},
		{toolCall("search-2", "web_search", map[string]any{"queries": []string{"second angle"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
	}
	results, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "storm impacts",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "storm impacts",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeBalanced,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results length = %d, want merged single URL: %#v", len(results), results)
	}
	if !strings.Contains(results[0].Content, "first angle") || !strings.Contains(results[0].Content, "second angle") {
		t.Fatalf("merged result content = %q, want both search angles", results[0].Content)
	}
}

func TestResearcherStopsAfterRepeatedSearchFailures(t *testing.T) {
	searcher := &alwaysFailingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"failed query 1"}})},
		{toolCall("search-2", "web_search", map[string]any{"queries": []string{"failed query 2"}})},
		{toolCall("search-3", "web_search", map[string]any{"queries": []string{"failed query 3"}})},
		{toolCall("search-4", "web_search", map[string]any{"queries": []string{"failed query 4"}})},
		{toolCall("search-5", "web_search", map[string]any{"queries": []string{"should not run"}})},
	}}
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
	}
	results, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "Harbin storm accidents",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "Harbin storm accidents",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err == nil {
		t.Fatal("Research error = nil, want search backend error")
	}
	if len(results) != 0 {
		t.Fatalf("results = %#v, want empty on repeated failures", results)
	}
	if len(searcher.queries) != maxFailedInformationToolCalls(ModeQuality) {
		t.Fatalf("queries = %#v, want %d failed attempts", searcher.queries, maxFailedInformationToolCalls(ModeQuality))
	}
	if len(researchModel.requests) > maxFailedInformationToolCalls(ModeQuality) {
		t.Fatalf("research model calls = %d, want capped after repeated failures", len(researchModel.requests))
	}
}

type staticTextModel struct {
	text string
}

type recordingSearchProvider struct {
	queries []string
}

func (p *recordingSearchProvider) Search(_ context.Context, query string, _ SearchOptions) ([]SearchResult, error) {
	p.queries = append(p.queries, query)
	return []SearchResult{{
		Title:   "\u54c8\u5c14\u6ee8\u5f3a\u5bf9\u6d41\u5929\u6c14",
		URL:     "https://example.com/harbin-tornado",
		Content: "2026\u5e745\u670831\u65e5\u54c8\u5c14\u6ee8\u51fa\u73b0\u5927\u98ce\u548c\u9f99\u5377\u98ce\u76f8\u5173\u62a5\u9053\u3002",
	}}, nil
}

type manySearchProvider struct{}

func (p *manySearchProvider) Search(_ context.Context, query string, opts SearchOptions) ([]SearchResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	results := make([]SearchResult, 0, limit)
	for i := 0; i < limit; i++ {
		results = append(results, SearchResult{
			Title:   fmt.Sprintf("Result %d", i+1),
			URL:     fmt.Sprintf("https://example.com/result-%d", i+1),
			Content: fmt.Sprintf("Search snippet %d for %s", i+1, query),
			Source:  opts.Source,
		})
	}
	return results, nil
}

type alwaysFailingSearchProvider struct {
	queries []string
}

func (p *alwaysFailingSearchProvider) Search(_ context.Context, query string, _ SearchOptions) ([]SearchResult, error) {
	p.queries = append(p.queries, query)
	return nil, errors.New("search backend unavailable")
}

type duplicateURLSearchProvider struct{}

func (p *duplicateURLSearchProvider) Search(_ context.Context, query string, opts SearchOptions) ([]SearchResult, error) {
	return []SearchResult{{
		Title:   "Storm report",
		URL:     "https://example.com/storm",
		Content: "snippet from " + query,
		Source:  opts.Source,
	}}, nil
}

func (m *staticTextModel) Info() model.Info { return model.Info{Name: "static"} }

func (m *staticTextModel) GenerateContent(_ context.Context, req *model.Request) (<-chan *model.Response, error) {
	if req == nil || len(req.Messages) == 0 {
		panic("classifier request missing messages")
	}
	if got := req.GenerationConfig.Stream; got {
		panic("classifier should not stream")
	}
	if !strings.Contains(req.Messages[0].Content, "search intent classifier") {
		panic("classifier system prompt missing")
	}
	out := make(chan *model.Response, 1)
	out <- &model.Response{
		Object:    model.ObjectTypeChatCompletion,
		Timestamp: time.Now(),
		Done:      true,
		Choices: []model.Choice{{
			Message: model.Message{Role: model.RoleAssistant, Content: m.text},
		}},
	}
	close(out)
	return out, nil
}

type scriptedResearchModel struct {
	calls    [][]model.ToolCall
	requests []*model.Request
}

func (m *scriptedResearchModel) Info() model.Info { return model.Info{Name: "scripted-research"} }

func (m *scriptedResearchModel) GenerateContent(_ context.Context, req *model.Request) (<-chan *model.Response, error) {
	m.requests = append(m.requests, req)
	out := make(chan *model.Response, 1)
	system := ""
	if req != nil && len(req.Messages) > 0 {
		system = req.Messages[0].Content
	}
	content := ""
	var toolCalls []model.ToolCall
	switch {
	case strings.Contains(system, "search result picker"):
		content = `{"picked_indices":[0]}`
	case strings.Contains(system, "information extractor"):
		content = `{"extracted_facts":"- Extracted tornado fact: outage and traffic disruption."}`
	default:
		if len(m.calls) > 0 {
			toolCalls = m.calls[0]
			m.calls = m.calls[1:]
		}
	}
	out <- &model.Response{
		Object:    model.ObjectTypeChatCompletion,
		Timestamp: time.Now(),
		Done:      true,
		Choices: []model.Choice{{
			Message: model.Message{Role: model.RoleAssistant, Content: content, ToolCalls: toolCalls},
		}},
	}
	close(out)
	return out, nil
}

func toolCall(id, name string, args map[string]any) model.ToolCall {
	payload, _ := json.Marshal(args)
	return model.ToolCall{
		ID:   id,
		Type: "function",
		Function: model.FunctionDefinitionParam{
			Name:      name,
			Arguments: payload,
		},
	}
}

type staticScrapeProvider struct {
	content string
}

func (p staticScrapeProvider) Scrape(_ context.Context, rawURL string) (ScrapedDocument, error) {
	return ScrapedDocument{URL: rawURL, Title: "Scraped", Content: p.content}, nil
}
