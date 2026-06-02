package vane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
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
	for _, query := range queries {
		if !containsCJK(query) {
			t.Fatalf("Chinese research query drifted to non-Chinese: %q in %#v", query, queries)
		}
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
	if got.StandaloneFollowUp != "那它们 benchmark 和用户反馈呢" {
		t.Fatalf("StandaloneFollowUp = %q, want latest Chinese query language preserved", got.StandaloneFollowUp)
	}
	if !hasSource(got.Sources, SearchSourceAcademic) || !hasSource(got.Sources, SearchSourceDiscussions) {
		t.Fatalf("sources = %#v, want academic and discussions", got.Sources)
	}
}

func TestClassifierPreservesChineseStandaloneFollowUp(t *testing.T) {
	classifier := &staticTextModel{text: `{
		"classification": {
			"skipSearch": false,
			"personalSearch": false,
			"academicSearch": false,
			"discussionSearch": false,
			"showWeatherWidget": false,
			"showStockWidget": false,
			"showCalculationWidget": false
		},
		"standaloneFollowUp": "What are the latest market reactions to NVIDIA's new AI PC hardware?"
	}`}
	agent := SearchAgent{ClassifierModel: classifier}
	query := "\u82f1\u4f1f\u8fbe\u63a8\u51fa\u7684\u65b0aipc\u786c\u4ef6 \u5e02\u573a\u53cd\u54cd\u5982\u4f55"
	got := agent.classify(context.Background(), SearchAgentRequest{
		Query:   query,
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if got.StandaloneFollowUp != query {
		t.Fatalf("StandaloneFollowUp = %q, want original Chinese query %q", got.StandaloneFollowUp, query)
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

func TestResearcherPromptStableAcrossIterations(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"vane search agent"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
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
	if len(researchModel.requests) < 2 {
		t.Fatalf("research requests = %d, want at least two", len(researchModel.requests))
	}
	firstPrompt := researchModel.requests[0].Messages[0].Content
	secondPrompt := researchModel.requests[1].Messages[0].Content
	if firstPrompt != secondPrompt {
		t.Fatalf("researcher system prompt changed across iterations")
	}
	if strings.Contains(firstPrompt, "Research iteration:") {
		t.Fatalf("researcher system prompt contains dynamic iteration text: %s", firstPrompt)
	}
	if !requestContainsContent(researchModel.requests[1], "<research_status iteration=\"2\"") {
		t.Fatalf("second request missing dynamic research status: %#v", researchModel.requests[1].Messages)
	}
}

func TestResearcherCompactsToolHistoryButKeepsFullResults(t *testing.T) {
	searcher := &recordingSearchProvider{content: strings.Repeat("full source content ", 80)}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"vane search agent"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
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
	if len(results) == 0 || len([]rune(results[0].Content)) <= historySnippetRunes {
		t.Fatalf("results lost full content: %#v", results)
	}
	if len(researchModel.requests) < 2 {
		t.Fatalf("research requests = %d, want at least two", len(researchModel.requests))
	}
	toolContent := lastToolMessageContent(researchModel.requests[1])
	if toolContent == "" {
		t.Fatalf("second request missing compacted tool message: %#v", researchModel.requests[1].Messages)
	}
	if strings.Contains(toolContent, strings.Repeat("full source content ", 40)) {
		t.Fatalf("tool history was not compacted: %d chars", len(toolContent))
	}
	if !strings.Contains(toolContent, "content") || !strings.Contains(toolContent, "...") {
		t.Fatalf("tool history missing compacted snippet: %s", toolContent)
	}
}

func TestResearcherPassesExtraFields(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query:       "what is vane",
		ExtraFields: map[string]any{"cache_prompt": true},
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
	if len(researchModel.requests) == 0 || researchModel.requests[0].ExtraFields["cache_prompt"] != true {
		t.Fatalf("research request extra fields = %#v, want cache_prompt", researchModel.requests)
	}
}

func TestAnsweringModelMergesDefaultExtraFields(t *testing.T) {
	base := &captureModel{}
	wrapped := &AnsweringModel{
		Base:        base,
		ExtraFields: map[string]any{"cache_prompt": true, "keep": "default"},
	}
	ch, err := wrapped.GenerateContent(context.Background(), &model.Request{
		Messages: []model.Message{model.NewUserMessage("hello")},
		ExtraFields: map[string]any{
			"keep":        "request",
			"temperature": 0,
		},
	})
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	for range ch {
	}
	if base.req == nil {
		t.Fatal("base model was not called")
	}
	if base.req.ExtraFields["cache_prompt"] != true ||
		base.req.ExtraFields["keep"] != "request" ||
		base.req.ExtraFields["temperature"] != 0 {
		t.Fatalf("extra fields = %#v, want merged defaults with request override", base.req.ExtraFields)
	}
}

func TestQualityResearchScrapesAndExtractsFacts(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"Harbin tornado official report"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	var events []SearchEvent
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
		ScrapeProvider: staticScrapeProvider{content: longArticleContent()},
		OnSearchEvent: func(_ context.Context, ev SearchEvent) {
			events = append(events, ev)
		},
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
	for _, phase := range []string{"picking_sources", "deep_read", "extract_facts"} {
		if !hasSearchEventPhase(events, phase) {
			t.Fatalf("events missing phase %q: %#v", phase, events)
		}
	}
}

func TestQualityResearchSkipsExtractorForShortScrapes(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"Harbin tornado official report"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	var events []SearchEvent
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
		ScrapeProvider: staticScrapeProvider{content: "Short report: tornado caused power outage and traffic disruption."},
		OnSearchEvent: func(_ context.Context, ev SearchEvent) {
			events = append(events, ev)
		},
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
	if len(results) == 0 || !strings.Contains(results[0].Content, "Short report") {
		t.Fatalf("quality results = %#v, want original short scraped content", results)
	}
	if hasSearchEventPhase(events, "extract_facts") {
		t.Fatalf("short scrape should not run extractor: %#v", events)
	}
	if countResearchRequestsContaining(researchModel.requests, "information extractor") != 0 {
		t.Fatalf("short scrape invoked extractor model")
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
		ScrapeProvider: staticScrapeProvider{content: longArticleContent()},
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

func TestExtractFactsCapsChunks(t *testing.T) {
	researchModel := &scriptedResearchModel{}
	researcher := Researcher{ResearchModel: researchModel}
	content := strings.Repeat("Harbin tornado damaged power lines and disrupted traffic. ", 700)
	got := researcher.extractFacts(context.Background(), ResearchRequest{
		Query:       "Harbin tornado impacts",
		Mode:        ModeQuality,
		Concurrency: 1,
	}, []string{"Harbin tornado impacts"}, content, "Long article")
	if !strings.Contains(got, "Extracted tornado fact") {
		t.Fatalf("extractFacts = %q, want extracted facts", got)
	}
	if got := countResearchRequestsContaining(researchModel.requests, "information extractor"); got != extractFactsMaxChunks {
		t.Fatalf("extractor calls = %d, want cap %d", got, extractFactsMaxChunks)
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

func TestQualityEventResultsFilterGenericSources(t *testing.T) {
	queries := []string{
		"\u54c8\u5c14\u6ee8 2026\u5e745\u670831\u65e5 \u5927\u98ce \u4e8b\u6545 \u5f71\u54cd",
		"\u54c8\u5c14\u6ee8 \u5f3a\u5bf9\u6d41 \u4ea4\u901a \u505c\u7535",
	}
	results := []SearchResult{
		{
			Title:   "\u54c8\u5c14\u6ee8\u906d\u5f3a\u5bf9\u6d41\u5929\u6c14\u4fb5\u88ad \u591a\u5730\u51fa\u73b0\u5927\u98ce\u6c99\u5c18",
			URL:     "https://news.qq.com/rain/a/20260531A07WZI00",
			Content: "5\u670831\u65e5\u54c8\u5c14\u6ee8\u5927\u98ce\u5bfc\u81f4\u4ea4\u901a\u53d7\u963b\u548c\u8bbe\u65bd\u53d7\u635f\u3002",
		},
		{
			Title:   "Harbin Travel Guide 2026",
			URL:     "https://www.chinahighlights.com/harbin/",
			Content: "Harbin facts, attractions, places to visit, and travel tips.",
		},
		{
			Title:   "\u54c8\u5c14\u6ee8_\u767e\u5ea6\u767e\u79d1",
			URL:     "https://baike.baidu.com/item/%E5%93%88%E5%B0%94%E6%BB%A8",
			Content: "\u54c8\u5c14\u6ee8\u662f\u9ed1\u9f99\u6c5f\u7701\u7701\u4f1a\u548c\u65c5\u6e38\u57ce\u5e02\u3002",
		},
		{
			Title:   "Why is Harbin not spelled Haerbin",
			URL:     "https://www.sohu.com/a/harbin-spelling",
			Content: "A language article about Harbin spelling.",
		},
	}
	got := filterQualityResults(queries, nil, results)
	if len(got) != 1 {
		t.Fatalf("filtered len = %d, want 1: %#v", len(got), got)
	}
	if got[0].URL != results[0].URL {
		t.Fatalf("kept result = %#v, want news result", got[0])
	}
}

func TestChineseVerboseSearchQueryIsRepaired(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{
		queryRepair: []string{
			"\u54c8\u5c14\u6ee8 2026\u5e745\u670831\u65e5 \u9f99\u5377\u98ce \u5b98\u65b9\u901a\u62a5",
			"\u54c8\u5c14\u6ee8 2026\u5e745\u670831\u65e5 \u5927\u98ce \u4e8b\u6545 \u4f24\u4ea1",
			"\u54c8\u5c14\u6ee8 5\u670831\u65e5 \u5f3a\u5bf9\u6d41 \u4ea4\u901a \u505c\u7535",
		},
		calls: [][]model.ToolCall{
			{toolCall("search-1", "web_search", map[string]any{"queries": []string{"\u641c\u7d22\u54c8\u5c14\u6ee82026\u5e745\u670831\u65e5\u7684\u5927\u98ce\uff0c\u770b\u770b\u51fa\u73b0\u4e86\u54ea\u4e9b\u91cd\u5927\u4e8b\u6545\uff0c\u5206\u6790\u5f71\u54cd"}})},
			{toolCall("done-1", "done", map[string]any{})},
		},
	}
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
	}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "\u641c\u7d22\u54c8\u5c14\u6ee8\u6628\u65e5\u7684\u5927\u98ce\uff0c\u770b\u770b\u51fa\u73b0\u4e86\u54ea\u4e9b\u91cd\u5927\u4e8b\u6545\uff0c\u5206\u6790\u4e00\u4e0b\u5e26\u6765\u4e86\u54ea\u4e9b\u5f71\u54cd",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "\u641c\u7d22\u54c8\u5c14\u6ee8\u6628\u65e5\u7684\u5927\u98ce\u4e8b\u6545\u53ca\u5f71\u54cd",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
		Now:     time.Date(2026, 6, 1, 10, 0, 0, 0, time.FixedZone("CST", 8*60*60)),
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(searcher.queries) != 3 {
		t.Fatalf("queries = %#v, want repaired 3 queries", searcher.queries)
	}
	for _, query := range searcher.queries {
		if looksLikeVerboseSearchQuery(query) {
			t.Fatalf("repaired query still verbose: %q", query)
		}
	}
}

func TestSearchPlannerSplitsChineseAnalysisGoal(t *testing.T) {
	searcher := &recordingSearchProvider{}
	classifier := &staticTextModel{text: `{
		"classification": {
			"skipSearch": false,
			"personalSearch": false,
			"academicSearch": false,
			"discussionSearch": false,
			"showWeatherWidget": false,
			"showStockWidget": false,
			"showCalculationWidget": false
		},
		"standaloneFollowUp": "帮我分析一下英伟达推出的新aipc硬件市场反响如何"
	}`}
	researchModel := &scriptedResearchModel{
		queryPlanRaw: `{
			"answer_goal": "分析英伟达新 AI PC 硬件的市场反响",
			"topic": "英伟达新 AI PC 硬件市场反响",
			"language": "zh",
			"report_sections": ["背景", "市场反馈", "影响"],
			"queries": [
				{"query": "英伟达 AI PC 硬件 市场反响", "purpose": "核心市场反馈", "source": "web", "priority": 1},
				{"query": "英伟达 AI PC 销量 评价", "purpose": "销量和用户评价", "source": "web", "priority": 2},
				{"query": "英伟达 AI PC 厂商 生态 影响", "purpose": "产业影响", "source": "web", "priority": 3}
			]
		}`,
		calls: [][]model.ToolCall{
			{toolCall("search-1", "web_search", map[string]any{"queries": []string{"帮我分析一下英伟达推出的新aipc硬件市场反响如何"}})},
			{toolCall("done-1", "done", map[string]any{})},
		},
	}
	agent := SearchAgent{ClassifierModel: classifier, ResearchModel: researchModel, SearchProvider: searcher}
	result, err := agent.Run(context.Background(), SearchAgentRequest{
		Query:   "帮我分析一下英伟达推出的新aipc硬件市场反响如何",
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.SearchPlan == nil || result.SearchPlan.AnswerGoal == "" || len(result.SearchPlan.Queries) < 3 {
		t.Fatalf("SearchPlan = %#v, want answer goal and planned queries", result.SearchPlan)
	}
	if len(searcher.queries) == 0 {
		t.Fatal("expected search queries")
	}
	for _, query := range searcher.queries {
		if hasTaskLanguage(query) || looksLikeVerboseSearchQuery(query) {
			t.Fatalf("query = %q, want planned keyword query; all=%#v", query, searcher.queries)
		}
		if !containsCJK(query) {
			t.Fatalf("query = %q, want Chinese query; all=%#v", query, searcher.queries)
		}
	}
}

func TestSearchPlannerFallbackForChineseReportGoal(t *testing.T) {
	plan := fallbackSearchPlanForQuery(
		"给我生成哈尔滨昨日大风事故影响分析报告",
		Classification{ShouldSearch: true, StandaloneFollowUp: "给我生成哈尔滨昨日大风事故影响分析报告"},
		ModeQuality,
		[]SearchSource{SearchSourceWeb},
		time.Date(2026, 6, 1, 10, 0, 0, 0, time.FixedZone("CST", 8*60*60)),
	)
	if len(plan.Queries) < 3 {
		t.Fatalf("queries = %#v, want several planned queries", plan.Queries)
	}
	for _, item := range plan.Queries {
		if hasTaskLanguage(item.Query) || strings.Contains(item.Query, "分析报告") {
			t.Fatalf("query = %q, leaked task language; plan=%#v", item.Query, plan)
		}
		if !containsCJK(item.Query) {
			t.Fatalf("query = %q, want Chinese query", item.Query)
		}
	}
}

func TestSearchPlannerFallbackForChineseIntroQuestion(t *testing.T) {
	plan := fallbackSearchPlanForQuery(
		"为我介绍-下什么是微软的WinML?",
		Classification{ShouldSearch: true, StandaloneFollowUp: "为我介绍-下什么是微软的WinML?"},
		ModeBalanced,
		[]SearchSource{SearchSourceWeb},
		time.Date(2026, 6, 2, 10, 0, 0, 0, time.FixedZone("CST", 8*60*60)),
	)
	if len(plan.Queries) == 0 {
		t.Fatalf("queries = %#v, want fallback queries", plan.Queries)
	}
	for _, item := range plan.Queries {
		if looksLikeDegenerateSearchQuery(item.Query) || hasTaskLanguage(item.Query) {
			t.Fatalf("query = %q, leaked task language; plan=%#v", item.Query, plan)
		}
		lowerQuery := strings.ToLower(item.Query)
		if !strings.Contains(lowerQuery, "winml") {
			t.Fatalf("query = %q, want WinML topic; plan=%#v", item.Query, plan)
		}
		if !containsCJK(item.Query) && !containsAnyFold(item.Query, "official documentation", "api reference", "github repository") {
			t.Fatalf("query = %q, want source-oriented WinML query; plan=%#v", item.Query, plan)
		}
	}
	return
	for _, item := range plan.Queries {
		if looksLikeDegenerateSearchQuery(item.Query) || hasTaskLanguage(item.Query) {
			t.Fatalf("query = %q, leaked task language; plan=%#v", item.Query, plan)
		}
		if !strings.Contains(item.Query, "微软") || !strings.Contains(strings.ToLower(item.Query), "winml") {
			t.Fatalf("query = %q, want Microsoft WinML topic; plan=%#v", item.Query, plan)
		}
	}
}

func TestSearchPlannerPrioritizesEnglishTechnicalQueriesForChineseWinMLPlan(t *testing.T) {
	rawQuery := "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML"
	plan := normalizeSearchPlan(SearchPlan{
		AnswerGoal: "\u6e05\u6670\u89e3\u91ca\u5fae\u8f6fWinML\u7684\u5b9a\u4e49\u3001\u6838\u5fc3\u529f\u80fd\u3001\u6280\u672f\u67b6\u6784\u3001\u9002\u7528\u573a\u666f\u53ca\u5176\u5728Windows\u751f\u6001\u4e2d\u7684\u89d2\u8272",
		Topic:      "\u5fae\u8f6f WinML \u6280\u672f",
		Language:   "zh",
		Queries: []PlannedSearchQuery{
			{Query: "\u5fae\u8f6f WinML \u5b9a\u4e49 \u6982\u8ff0", Source: SearchSourceWeb, Priority: 1},
			{Query: "WinML \u6280\u672f\u67b6\u6784 \u5de5\u4f5c\u539f\u7406", Source: SearchSourceWeb, Priority: 2},
			{Query: "WinML \u5e94\u7528\u573a\u666f \u6848\u4f8b", Source: SearchSourceWeb, Priority: 3},
		},
	}, rawQuery, Classification{ShouldSearch: true}, ModeQuality, []SearchSource{SearchSourceWeb}, time.Date(2026, 6, 2, 10, 0, 0, 0, time.FixedZone("CST", 8*60*60)))

	want := []string{
		"WinML official documentation",
		"WinML API reference",
		"WinML GitHub repository",
	}
	if len(plan.Queries) < len(want) {
		t.Fatalf("queries=%#v, want technical query overrides", plan.Queries)
	}
	for i, query := range want {
		if plan.Queries[i].Query != query {
			t.Fatalf("query[%d]=%q, want %q; all=%#v", i, plan.Queries[i].Query, query, plan.Queries)
		}
	}
}

func TestResearcherReplacesDegenerateToolQueryWithSearchPlan(t *testing.T) {
	searcher := &recordingSearchProvider{}
	plan := &SearchPlan{Queries: []PlannedSearchQuery{
		{Query: "微软 WinML Windows ML", Source: SearchSourceWeb, Priority: 1},
		{Query: "WinML Windows Machine Learning", Source: SearchSourceWeb, Priority: 2},
	}}
	researchModel := &scriptedResearchModel{
		calls: [][]model.ToolCall{
			{toolCall("search-1", "web_search", map[string]any{"queries": []string{"为"}})},
			{toolCall("done-1", "done", map[string]any{})},
		},
	}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "为我介绍-下什么是微软的WinML?",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "为我介绍-下什么是微软的WinML?",
			SearchPlan:         plan,
			Sources:            []SearchSource{SearchSourceWeb},
		},
		SearchPlan: plan,
		Mode:       ModeBalanced,
		Sources:    []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(searcher.queries) == 0 {
		t.Fatalf("queries=%#v, want planned queries", searcher.queries)
	}
	for _, query := range searcher.queries {
		if looksLikeDegenerateSearchQuery(query) {
			t.Fatalf("query = %q, want planned query; all=%#v", query, searcher.queries)
		}
		if !strings.Contains(strings.ToLower(query), "winml") && !strings.Contains(strings.ToLower(query), "windows machine learning") {
			t.Fatalf("query = %q, want WinML planned query; all=%#v", query, searcher.queries)
		}
	}
}

func TestDeterministicResearchUsesSearchPlanQueries(t *testing.T) {
	searcher := &recordingSearchProvider{}
	plan := &SearchPlan{Queries: []PlannedSearchQuery{
		{Query: "英伟达 AI PC 硬件 市场反响", Source: SearchSourceWeb, Priority: 1},
		{Query: "英伟达 AI PC 销量 评价", Source: SearchSourceWeb, Priority: 2},
	}}
	researcher := Researcher{SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "帮我分析一下英伟达推出的新aipc硬件市场反响如何",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "帮我分析一下英伟达推出的新aipc硬件市场反响如何",
			SearchPlan:         plan,
			Sources:            []SearchSource{SearchSourceWeb},
		},
		SearchPlan: plan,
		Mode:       ModeBalanced,
		Sources:    []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(searcher.queries) != 2 {
		t.Fatalf("queries=%#v, want planned queries", searcher.queries)
	}
	for _, query := range searcher.queries {
		if hasTaskLanguage(query) {
			t.Fatalf("query = %q, leaked task language", query)
		}
	}
}

func TestResearcherLexicalRankingFiltersOffTopicTechnicalResults(t *testing.T) {
	researcher := Researcher{}
	queries := []string{
		"WinML main scenarios",
		"WinML architecture components",
		"WinML developer docs examples",
	}
	results := []SearchResult{
		{
			Title:   "Machine learning main scenarios",
			URL:     "https://example.com/ml-scenarios",
			Content: "A generic article about machine learning scenarios.",
			Query:   queries[0],
		},
		{
			Title:   "Windows ML samples and documentation",
			URL:     "https://learn.microsoft.com/windows/ai/windows-ml/samples",
			Content: "Windows Machine Learning is also known as WinML.",
			Query:   queries[2],
		},
		{
			Title:   "Using WinML from C#",
			URL:     "https://developer.example.com/winml-csharp",
			Content: "WinML application guidance.",
			Query:   queries[0],
		},
	}

	ranked := researcher.rankAndDedupe(context.Background(), queries, []string{"WinML"}, results, ModeBalanced)
	if len(ranked) != 2 {
		t.Fatalf("ranked results = %#v, want only WinML-related results", ranked)
	}
	if ranked[0].URL != "https://learn.microsoft.com/windows/ai/windows-ml/samples" {
		t.Fatalf("top result = %#v, want Microsoft documentation first", ranked[0])
	}
	for _, result := range ranked {
		if strings.Contains(result.URL, "ml-scenarios") {
			t.Fatalf("off-topic result was not filtered: %#v", ranked)
		}
	}
}

func TestChineseSearchQueriesRejectEnglishRepair(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{
		queryRepair: []string{
			"Harbin May 31 2026 tornado official report",
			"Harbin storm accident casualties",
			"Harbin severe weather impact traffic power outage",
		},
		calls: [][]model.ToolCall{
			{toolCall("search-1", "web_search", map[string]any{"queries": []string{"搜索哈尔滨2026年5月31日的大风，看看出现了哪些重大事故，分析影响"}})},
			{toolCall("done-1", "done", map[string]any{})},
		},
	}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "搜索哈尔滨昨日的大风，看看出现了哪些重大事故，分析一下带来了哪些影响",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "搜索哈尔滨昨日的大风事故及影响",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
		Now:     time.Date(2026, 6, 1, 10, 0, 0, 0, time.FixedZone("CST", 8*60*60)),
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(searcher.queries) == 0 {
		t.Fatal("expected fallback Chinese search queries")
	}
	for _, query := range searcher.queries {
		if !containsCJK(query) {
			t.Fatalf("query = %q, want Chinese fallback query; all=%#v", query, searcher.queries)
		}
		if containsAnyFold(query, "Harbin", "tornado", "official report", "casualties", "power outage") {
			t.Fatalf("query = %q, leaked English repaired query; all=%#v", query, searcher.queries)
		}
	}
}

func TestChineseSearchQueriesRejectEnglishToolQueries(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{
			"NVIDIA AI PC hardware market reaction 2026",
			"NVIDIA RTX AI PC sales reviews",
			"NVIDIA AI PC launch feedback",
		}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	query := "\u82f1\u4f1f\u8fbe\u63a8\u51fa\u7684\u65b0aipc\u786c\u4ef6 \u5e02\u573a\u53cd\u54cd\u5982\u4f55"
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: query,
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "What are the latest market reactions to NVIDIA's new AI PC hardware?",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(searcher.queries) == 0 {
		t.Fatal("expected Chinese fallback search queries")
	}
	for _, query := range searcher.queries {
		if !containsCJK(query) {
			t.Fatalf("query = %q, want Chinese fallback query; all=%#v", query, searcher.queries)
		}
		if containsAnyFold(query, "NVIDIA AI PC", "market reaction", "launch feedback") {
			t.Fatalf("query = %q, leaked English tool query; all=%#v", query, searcher.queries)
		}
	}
}

func TestChineseTechnicalSearchAllowsEnglishEntityQueries(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{
			"WinML API reference Microsoft",
			"WinML ONNX model inference Windows",
			"WinML DirectML GPU acceleration",
		}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	want := []string{
		"WinML API reference Microsoft",
		"WinML ONNX model inference Windows",
		"WinML DirectML GPU acceleration",
	}
	got := map[string]bool{}
	for _, query := range searcher.queries {
		got[query] = true
	}
	for _, query := range want {
		if !got[query] {
			t.Fatalf("queries=%#v, want English technical query %q", searcher.queries, query)
		}
	}
	if len(searcher.queries) != len(want) {
		t.Fatalf("queries=%#v, want English technical queries %#v", searcher.queries, want)
	}
}

func TestChineseTechnicalSearchKeepsOfficialSiteQueries(t *testing.T) {
	searcher := &recordingSearchProvider{}
	plan := &SearchPlan{
		AnswerGoal: "\u89e3\u91ca\u5fae\u8f6fWinML\u7684\u5b9a\u4e49\u3001API\u3001ONNX\u63a8\u7406\u548c\u786c\u4ef6\u52a0\u901f",
		Topic:      "\u5fae\u8f6f WinML",
		Queries: []PlannedSearchQuery{
			{Query: "WinML API reference Microsoft", Source: SearchSourceWeb, Priority: 1},
			{Query: "WinML Windows Machine Learning official documentation", Source: SearchSourceWeb, Priority: 2},
		},
	}
	toolQueries := []string{
		"Windows ML API reference site:learn.microsoft.com",
		"WinML ONNX runtime inference site:github.com/microsoft/Windows-ML",
		"Windows Machine Learning hardware acceleration NPU GPU CPU",
	}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": toolQueries})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
			SearchPlan:         plan,
			Sources:            []SearchSource{SearchSourceWeb},
		},
		SearchPlan: plan,
		Mode:       ModeQuality,
		Sources:    []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if !sameStringSet(searcher.queries, toolQueries) {
		t.Fatalf("queries=%#v, want tool site queries %#v", searcher.queries, toolQueries)
	}
}

func TestChineseTechnicalSearchDoesNotRepairPlanBackToChinese(t *testing.T) {
	searcher := &recordingSearchProvider{}
	plan := &SearchPlan{
		AnswerGoal: "\u89e3\u91ca\u5fae\u8f6fWinML\u7684\u5b9a\u4e49\u3001API\u3001ONNX\u63a8\u7406\u548c\u786c\u4ef6\u52a0\u901f",
		Topic:      "\u5fae\u8f6f WinML",
		Queries: []PlannedSearchQuery{
			{Query: "WinML API reference Microsoft", Source: SearchSourceWeb, Priority: 1},
			{Query: "WinML Windows Machine Learning official documentation", Source: SearchSourceWeb, Priority: 2},
			{Query: "WinML ONNX model inference Windows", Source: SearchSourceWeb, Priority: 3},
			{Query: "\u5fae\u8f6f WinML \u4ecb\u7ecd", Source: SearchSourceWeb, Priority: 4},
		},
	}
	researchModel := &scriptedResearchModel{
		queryRepair: []string{
			"\u5fae\u8f6f WinML",
			"WinML Windows \u673a\u5668\u5b66\u4e60 API",
			"WinML \u5b98\u65b9\u6587\u6863 \u529f\u80fd",
		},
		calls: [][]model.ToolCall{
			{toolCall("search-1", "web_search", map[string]any{"queries": []string{"\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML"}})},
			{toolCall("done-1", "done", map[string]any{})},
		},
	}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
			SearchPlan:         plan,
			Sources:            []SearchSource{SearchSourceWeb},
		},
		SearchPlan: plan,
		Mode:       ModeQuality,
		Sources:    []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	want := []string{
		"WinML API reference Microsoft",
		"WinML Windows Machine Learning official documentation",
		"WinML ONNX model inference Windows",
	}
	if !sameStringSet(searcher.queries, want) {
		t.Fatalf("queries=%#v, want unrepaired technical plan queries %#v", searcher.queries, want)
	}
	if countResearchRequestsContaining(researchModel.requests, "search query repairer") != 0 {
		t.Fatalf("technical plan queries should not invoke repairer")
	}
}

func TestFinalTechnicalResultsFilterLowValueSearchPages(t *testing.T) {
	req := ResearchRequest{
		Query: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
		SearchPlan: &SearchPlan{Entities: []string{"WinML"}, Queries: []PlannedSearchQuery{
			{Query: "WinML API reference Microsoft", Source: SearchSourceWeb},
			{Query: "WinML ONNX model inference Windows", Source: SearchSourceWeb},
		}},
	}
	ranked := rankFinalResearchResults(req, []SearchResult{
		{
			Query:   "WinML API \u529f\u80fd \u7279\u6027",
			Title:   "winml api \u529f\u80fd \u7279\u6027 - 360\u6587\u5e93",
			URL:     "https://wenku.so.com/s?q=winml%20api",
			Content: "short search landing page",
		},
		{
			Query:   "\u5fae\u8f6f\u7684WinML",
			Title:   "\u5fae\u8f6f_\u767e\u5ea6\u767e\u79d1",
			URL:     "https://baike.baidu.com/item/%E5%BE%AE%E8%BD%AF/124767",
			Content: "\u5fae\u8f6f\u516c\u53f8\u4ecb\u7ecd",
		},
		{
			Title:   "Windows AI | Microsoft Learn",
			URL:     "https://learn.microsoft.com/en-us/windows/ai/",
			Content: "Windows Machine Learning and WinML guidance for local AI inference.",
		},
		{
			Title:   "GitHub - microsoft/WindowsML",
			URL:     "https://github.com/microsoft/WindowsML",
			Content: "Official repo for Windows ML, Microsoft's high-performance local AI inferencing framework for Windows.",
		},
	})
	if len(ranked) != 2 {
		t.Fatalf("ranked=%#v, want only official technical sources", ranked)
	}
	for _, result := range ranked {
		if strings.Contains(result.URL, "wenku.so.com") || strings.Contains(result.URL, "baike.baidu.com") {
			t.Fatalf("low-value source survived: %#v", ranked)
		}
	}
	if !strings.Contains(ranked[0].URL, "learn.microsoft.com") && !strings.Contains(ranked[0].URL, "github.com/microsoft") {
		t.Fatalf("top result = %#v, want official technical source", ranked[0])
	}
}

func TestFinalTechnicalResultsRequirePrimaryEntityMatch(t *testing.T) {
	req := ResearchRequest{
		Query: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
		SearchPlan: &SearchPlan{Queries: []PlannedSearchQuery{
			{Query: "WinML API reference Microsoft", Source: SearchSourceWeb},
			{Query: "WinML ONNX model inference Windows", Source: SearchSourceWeb},
		}, Entities: []string{"WinML"}},
	}
	ranked := rankFinalResearchResults(req, []SearchResult{
		{
			Title:   "Download Windows 11",
			URL:     "https://www.microsoft.com/software-download/windows11",
			Content: "Download the latest Windows installer and media creation tool.",
		},
		{
			Title:   "DiskGenius official download",
			URL:     "https://www.diskgenius.com/download.php",
			Content: "Disk partition management and file recovery software.",
		},
		{
			Title:   "Windows app development docs",
			URL:     "https://learn.microsoft.com/en-us/windows/apps/",
			Content: "Documentation for Windows application development.",
		},
	})
	if len(ranked) != 0 {
		t.Fatalf("ranked=%#v, want no sources without WinML primary entity match", ranked)
	}
}

func TestQualityDeepReadRequiresPrimaryEntityMatch(t *testing.T) {
	researcher := Researcher{
		ScrapeProvider: staticScrapeProvider{content: "DiskGenius partition recovery details."},
	}
	req := ResearchRequest{
		Query: "\u4e3a\u6211\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u5fae\u8f6f\u7684WinML",
		Mode:  ModeQuality,
		SearchPlan: &SearchPlan{
			Entities: []string{"WinML"},
			Queries:  []PlannedSearchQuery{{Query: "WinML API reference Microsoft", Source: SearchSourceWeb}},
		},
	}
	results := []SearchResult{
		{
			Query:   "WinML API reference Microsoft",
			Title:   "Download Windows 11",
			URL:     "https://www.microsoft.com/software-download/windows11",
			Content: "Download the latest Windows installer.",
		},
		{
			Query:   "WinML API reference Microsoft",
			Title:   "DiskGenius official download",
			URL:     "https://www.diskgenius.com/download.php",
			Content: "Disk partition management and file recovery software.",
		},
	}
	got := researcher.deepReadQuality(context.Background(), req, []string{"WinML API reference Microsoft"}, results)
	if len(got) != 0 {
		t.Fatalf("deepReadQuality=%#v, want empty when no source matches primary entity", got)
	}
}

func TestGoodKeywordQueriesDoNotInvokeRepair(t *testing.T) {
	searcher := &recordingSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"\u54c8\u5c14\u6ee8 5\u670831\u65e5 \u9f99\u5377\u98ce"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{ResearchModel: researchModel, SearchProvider: searcher}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "\u54c8\u5c14\u6ee8\u9f99\u5377\u98ce",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "\u54c8\u5c14\u6ee8\u9f99\u5377\u98ce",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(searcher.queries) != 1 || searcher.queries[0] != "\u54c8\u5c14\u6ee8 5\u670831\u65e5 \u9f99\u5377\u98ce" {
		t.Fatalf("queries=%#v, want original keyword query", searcher.queries)
	}
	for _, req := range researchModel.requests {
		if len(req.Messages) > 0 && strings.Contains(req.Messages[0].Content, "search query repairer") {
			t.Fatalf("repair model invoked for good keyword query")
		}
	}
}

func TestWebSearchActionDescriptionMatchesOriginalQueryGuidance(t *testing.T) {
	desc := webSearchActionDescription(ModeQuality)
	for _, want := range []string{
		"Your queries should be very targeted and specific",
		"Your queries shouldn't be sentences",
		"SEO friendly",
		"Start initially with broader queries",
		"Do not stop before at least 3 information attempts",
	} {
		if !strings.Contains(desc, want) {
			t.Fatalf("web search description missing %q: %s", want, desc)
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
	})
	if !strings.Contains(prompt, "aim for 3-5 information-gathering calls") ||
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
	})
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
}

func TestResearcherExecutesSearchQueriesConcurrently(t *testing.T) {
	searcher := &blockingSearchProvider{gate: make(chan struct{})}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"one", "two", "three"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
		Concurrency:    3,
	}
	done := make(chan error, 1)
	go func() {
		_, err := researcher.Research(context.Background(), ResearchRequest{
			Query: "parallel search",
			Classification: Classification{
				ShouldSearch:       true,
				StandaloneFollowUp: "parallel search",
				Sources:            []SearchSource{SearchSourceWeb},
			},
			Mode:    ModeQuality,
			Sources: []SearchSource{SearchSourceWeb},
		})
		done <- err
	}()
	deadline := time.After(time.Second)
	for searcher.activeCount() < 3 {
		select {
		case <-deadline:
			t.Fatalf("active searches = %d, want 3 concurrent searches", searcher.activeCount())
		default:
			time.Sleep(5 * time.Millisecond)
		}
	}
	close(searcher.gate)
	if err := <-done; err != nil {
		t.Fatalf("Research: %v", err)
	}
}

func TestSoftInformationBudgetRequiresUsefulQualityResults(t *testing.T) {
	generic := []SearchResult{
		{Title: "哈尔滨市_百度百科", URL: "https://baike.baidu.com/item/harbin", Content: strings.Repeat("百科", 20)},
		{Title: "哈尔滨旅游攻略", URL: "https://example.com/travel", Content: strings.Repeat("旅游", 20)},
		{Title: "哈尔滨市人民政府", URL: "https://www.harbin.gov.cn/", Content: strings.Repeat("首页", 20)},
	}
	if shouldStopAfterSoftInformationBudget(ModeQuality, generic, 1, 1) {
		t.Fatal("quality soft budget should not stop on generic or short results")
	}
	var useful []SearchResult
	for i := 0; i < 12; i++ {
		useful = append(useful, SearchResult{
			Title:   fmt.Sprintf("Harbin storm report %d", i),
			URL:     fmt.Sprintf("https://news.example.com/%d", i),
			Content: strings.Repeat("Harbin tornado storm outage traffic damage. ", 30),
		})
	}
	if !shouldStopAfterSoftInformationBudget(ModeQuality, useful, 1, 1) {
		t.Fatal("quality soft budget should stop after enough useful long-form results")
	}
}

func TestResearcherDoesNotSoftStopOnGenericQualityResults(t *testing.T) {
	searcher := &genericSearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"Harbin storm"}})},
		{toolCall("search-2", "web_search", map[string]any{"queries": []string{"Harbin storm damage"}})},
		{toolCall("done-1", "done", map[string]any{})},
	}}
	researcher := Researcher{
		ResearchModel:  researchModel,
		SearchProvider: searcher,
	}
	_, err := researcher.Research(context.Background(), ResearchRequest{
		Query: "storm impact",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "storm impact",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:                    ModeQuality,
		Sources:                 []SearchSource{SearchSourceWeb},
		SoftMaxInformationCalls: 1,
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(researchModel.requests) < 2 {
		t.Fatalf("research model requests = %d, want no soft stop on generic/short quality results", len(researchModel.requests))
	}
}

func TestResearcherToolLoopDoesNotStopAtResultLimit(t *testing.T) {
	searcher := &queryScopedManySearchProvider{}
	researchModel := &scriptedResearchModel{calls: [][]model.ToolCall{
		{toolCall("search-1", "web_search", map[string]any{"queries": []string{"first angle", "first detail", "first source"}})},
		{toolCall("search-2", "web_search", map[string]any{"queries": []string{"second angle", "second detail", "second source"}})},
		{toolCall("done-1", "done", map[string]any{})},
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
		Query: "wide research",
		Classification: Classification{
			ShouldSearch:       true,
			StandaloneFollowUp: "wide research",
			Sources:            []SearchSource{SearchSourceWeb},
		},
		Mode:    ModeQuality,
		Sources: []SearchSource{SearchSourceWeb},
	})
	if err != nil {
		t.Fatalf("Research: %v", err)
	}
	if len(researchModel.requests) != 3 {
		t.Fatalf("research model requests = %d, want second search plus done after reaching source limit", len(researchModel.requests))
	}
	if len(results) <= totalResultLimit(ModeQuality) {
		t.Fatalf("results = %d, want researcher to keep gathering beyond final context limit", len(results))
	}
	end := lastSearchEventOfType(events, SearchEventEnd)
	if end == nil {
		t.Fatalf("missing end event: %#v", events)
	}
	if got, _ := end.Metadata["stop_reason"].(string); got != "done" {
		t.Fatalf("stop_reason = %q, want done; metadata=%#v", got, end.Metadata)
	}
	if len(end.Results) != totalResultLimit(ModeQuality) {
		t.Fatalf("end results = %d, want final event clipped to context limit %d", len(end.Results), totalResultLimit(ModeQuality))
	}
}

type staticTextModel struct {
	text string
}

type recordingSearchProvider struct {
	mu      sync.Mutex
	queries []string
	content string
}

func (p *recordingSearchProvider) Search(_ context.Context, query string, _ SearchOptions) ([]SearchResult, error) {
	p.mu.Lock()
	p.queries = append(p.queries, query)
	p.mu.Unlock()
	return []SearchResult{{
		Title:   "\u54c8\u5c14\u6ee8\u5f3a\u5bf9\u6d41\u5929\u6c14",
		URL:     "https://example.com/harbin-tornado",
		Content: firstNonEmpty(p.content, "2026\u5e745\u670831\u65e5\u54c8\u5c14\u6ee8\u51fa\u73b0\u5927\u98ce\u548c\u9f99\u5377\u98ce\u76f8\u5173\u62a5\u9053\u3002"),
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

type queryScopedManySearchProvider struct{}

func (p *queryScopedManySearchProvider) Search(_ context.Context, query string, opts SearchOptions) ([]SearchResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	results := make([]SearchResult, 0, limit)
	prefix := strings.NewReplacer(" ", "-", "/", "-", "?", "").Replace(strings.ToLower(query))
	for i := 0; i < limit; i++ {
		results = append(results, SearchResult{
			Title:   fmt.Sprintf("%s result %d", query, i+1),
			URL:     fmt.Sprintf("https://example.com/%s/result-%d", prefix, i+1),
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

type genericSearchProvider struct{}

func (p *genericSearchProvider) Search(_ context.Context, query string, opts SearchOptions) ([]SearchResult, error) {
	return []SearchResult{
		{Title: "哈尔滨市_百度百科", URL: "https://baike.baidu.com/item/harbin", Content: "哈尔滨百科", Source: opts.Source},
		{Title: "哈尔滨旅游攻略", URL: "https://example.com/travel", Content: "哈尔滨旅游景点", Source: opts.Source},
		{Title: "哈尔滨市人民政府", URL: "https://www.harbin.gov.cn/", Content: "政府首页", Source: opts.Source},
	}, nil
}

type blockingSearchProvider struct {
	mu     sync.Mutex
	active int
	gate   chan struct{}
}

func (p *blockingSearchProvider) Search(_ context.Context, query string, opts SearchOptions) ([]SearchResult, error) {
	p.mu.Lock()
	p.active++
	p.mu.Unlock()
	<-p.gate
	return []SearchResult{{
		Title:   query,
		URL:     "https://example.com/" + query,
		Content: "result for " + query,
		Source:  opts.Source,
	}}, nil
}

func (p *blockingSearchProvider) activeCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.active
}

func hasSearchEventPhase(events []SearchEvent, phase string) bool {
	for _, ev := range events {
		if ev.Type != SearchEventResearchStep || ev.Metadata == nil {
			continue
		}
		if got, _ := ev.Metadata["phase"].(string); got == phase {
			return true
		}
	}
	return false
}

func countResearchRequestsContaining(requests []*model.Request, needle string) int {
	count := 0
	for _, req := range requests {
		if req == nil || len(req.Messages) == 0 {
			continue
		}
		if strings.Contains(req.Messages[0].Content, needle) {
			count++
		}
	}
	return count
}

func sameStringSet(left []string, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	counts := map[string]int{}
	for _, item := range left {
		counts[item]++
	}
	for _, item := range right {
		counts[item]--
		if counts[item] < 0 {
			return false
		}
	}
	return true
}

func requestContainsContent(req *model.Request, needle string) bool {
	if req == nil {
		return false
	}
	for _, msg := range req.Messages {
		if strings.Contains(msg.Content, needle) {
			return true
		}
	}
	return false
}

func lastToolMessageContent(req *model.Request) string {
	if req == nil {
		return ""
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == model.RoleTool {
			return req.Messages[i].Content
		}
	}
	return ""
}

func longArticleContent() string {
	return strings.Repeat("Long article: tornado caused power outage and traffic disruption. ", 30)
}

func lastSearchEventOfType(events []SearchEvent, typ SearchEventType) *SearchEvent {
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Type == typ {
			return &events[i]
		}
	}
	return nil
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
	calls          [][]model.ToolCall
	requests       []*model.Request
	queryPlanRaw   string
	queryRepair    []string
	queryRepairRaw string
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
	case strings.Contains(system, "search query planner"):
		content = firstNonEmpty(m.queryPlanRaw, `{"answer_goal":"test goal","topic":"test topic","language":"en","queries":[{"query":"test topic","source":"web","priority":1}]}`)
	case strings.Contains(system, "search query repairer"):
		if m.queryRepairRaw != "" {
			content = m.queryRepairRaw
		} else if len(m.queryRepair) > 0 {
			payload, _ := json.Marshal(map[string][]string{"queries": m.queryRepair})
			content = string(payload)
		} else {
			content = `{"queries":[]}`
		}
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
