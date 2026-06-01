package vane

import (
	"context"
	"strings"
	"testing"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

func TestClassifierSkipsConversationalQuery(t *testing.T) {
	got := classifySearch(SearchAgentRequest{
		Query:   "你好",
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
		{name: "weather", text: "上海今天的天气怎么样", want: func(c Classification) bool { return c.NeedWeather }},
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

func TestResearchQueriesResolveHarbinYesterdayWindQuestion(t *testing.T) {
	now := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	query := "搜索哈尔滨昨日的大风，看看出现了哪些重大事故，分析一下带来了哪些影响"
	queries := buildResearchQueries(query, ModeQuality, 25, now)
	if len(queries) == 0 {
		t.Fatal("expected research queries")
	}
	first := queries[0]
	if strings.Contains(first, "昨日") || strings.Contains(first, "昨天") {
		t.Fatalf("query still contains relative date: %q", first)
	}
	if !strings.Contains(first, "2026年5月31日") {
		t.Fatalf("query = %q, want absolute date 2026年5月31日", first)
	}
	if !strings.Contains(first, "哈尔滨") || !strings.Contains(first, "大风") {
		t.Fatalf("query = %q, want Harbin wind terms", first)
	}
}

func TestSearchAgentHarbinQuestionUsesAbsoluteDateQueries(t *testing.T) {
	now := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	searcher := &recordingSearchProvider{}
	agent := SearchAgent{SearchProvider: searcher}
	_, err := agent.Run(context.Background(), SearchAgentRequest{
		Query:         "搜索哈尔滨昨日的大风，看看出现了哪些重大事故，分析一下带来了哪些影响",
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
		if strings.Contains(query, "昨日") || strings.Contains(query, "昨天") {
			t.Fatalf("search query still contains relative date: %q", query)
		}
	}
	if !strings.Contains(searcher.queries[0], "2026年5月31日") {
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
		Query:    "上海天气",
		Messages: []model.Message{model.NewUserMessage("上海天气")},
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

type staticTextModel struct {
	text string
}

type recordingSearchProvider struct {
	queries []string
}

func (p *recordingSearchProvider) Search(_ context.Context, query string, _ SearchOptions) ([]SearchResult, error) {
	p.queries = append(p.queries, query)
	return []SearchResult{{
		Title:   "哈尔滨强对流天气",
		URL:     "https://example.com/harbin-tornado",
		Content: "2026年5月31日哈尔滨出现大风和龙卷风相关报道。",
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
