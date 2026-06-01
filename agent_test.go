package vane

import (
	"context"
	"testing"

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
			queries := buildResearchQueries("vane", tt.mode, defaultMaxIterations(tt.mode))
			if len(queries) != tt.want {
				t.Fatalf("query count = %d, want %d: %#v", len(queries), tt.want, queries)
			}
		})
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
