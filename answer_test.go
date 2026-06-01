package vane

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

type fakeSearchProvider struct {
	queries []string
}

func (f *fakeSearchProvider) Search(_ context.Context, query string, _ SearchOptions) ([]SearchResult, error) {
	f.queries = append(f.queries, query)
	return []SearchResult{
		{Title: "First", URL: "https://example.com/a#x", Content: "Alpha fact"},
		{Title: "Duplicate", URL: "https://example.com/a", Content: "Duplicate fact"},
		{Title: "Second", URL: "https://example.com/b", Content: "Beta fact"},
	}, nil
}

type failingSearchProvider struct {
	queries []string
}

func (f *failingSearchProvider) Search(_ context.Context, query string, _ SearchOptions) ([]SearchResult, error) {
	f.queries = append(f.queries, query)
	return nil, errors.New("search backend unavailable")
}

type captureModel struct {
	req *model.Request
}

func (m *captureModel) Info() model.Info { return model.Info{Name: "capture"} }

func (m *captureModel) GenerateContent(_ context.Context, req *model.Request) (<-chan *model.Response, error) {
	m.req = req
	out := make(chan *model.Response, 1)
	out <- &model.Response{
		Object:    model.ObjectTypeChatCompletion,
		Timestamp: time.Now(),
		Done:      true,
		Choices: []model.Choice{{
			Message: model.Message{Role: model.RoleAssistant, Content: "ok"},
		}},
	}
	close(out)
	return out, nil
}

func TestAnswerUsesSearchResultsInWriterPrompt(t *testing.T) {
	searcher := &fakeSearchProvider{}
	base := &captureModel{}
	ch, err := Answer(context.Background(), Request{
		Model:          base,
		SearchProvider: searcher,
		Mode:           ModeBalanced,
		Messages: []model.Message{
			model.NewUserMessage("what is vane"),
		},
		Now: time.Date(2026, 5, 31, 0, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	for range ch {
	}
	if len(searcher.queries) != 6 {
		t.Fatalf("queries = %v, want 6 balanced research queries", searcher.queries)
	}
	if base.req == nil || len(base.req.Messages) == 0 {
		t.Fatal("base model did not receive writer request")
	}
	prompt := base.req.Messages[0].Content
	if !strings.Contains(prompt, "Alpha fact") || !strings.Contains(prompt, "[number]") {
		t.Fatalf("writer prompt missing source context/citation instruction: %s", prompt)
	}
	if !strings.Contains(prompt, "Every sentence in the response that relies on web context should include at least one citation") {
		t.Fatalf("writer prompt missing original-style per-sentence citation instruction: %s", prompt)
	}
	if strings.Count(prompt, `https://example.com/a`) != 1 {
		t.Fatalf("duplicate URL was not deduplicated in prompt: %s", prompt)
	}
}

func TestQualityWriterPromptMatchesOriginalDepthRequirement(t *testing.T) {
	prompt := getWriterPrompt("<search_results><result index=\"1\">Fact</result></search_results>", "", "", ModeQuality, time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC))
	if !strings.Contains(prompt, "at least 2000 words") ||
		!strings.Contains(prompt, "frame the answer like a research report") ||
		!strings.Contains(prompt, "do not add a main title") {
		t.Fatalf("quality writer prompt missing original depth/format requirements: %s", prompt)
	}
}

func TestAnswerUsesWriterPromptWhenSearchFails(t *testing.T) {
	searcher := &failingSearchProvider{}
	base := &captureModel{}
	ch, err := Answer(context.Background(), Request{
		Model:          base,
		SearchProvider: searcher,
		Mode:           ModeSpeed,
		Messages: []model.Message{
			model.NewUserMessage("search yesterday storm impact"),
		},
		Now: time.Date(2026, 6, 1, 10, 0, 0, 0, time.FixedZone("CST", 8*60*60)),
	})
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	for range ch {
	}
	if len(searcher.queries) == 0 {
		t.Fatal("expected search to be attempted")
	}
	if base.req == nil || len(base.req.Messages) == 0 {
		t.Fatal("base model did not receive writer request")
	}
	prompt := base.req.Messages[0].Content
	if !strings.Contains(prompt, "<no_results>") {
		t.Fatalf("writer prompt missing no-results guard: %s", prompt)
	}
	if !strings.Contains(prompt, "2026-06-01T10:00:00+08:00") {
		t.Fatalf("writer prompt missing local current date: %s", prompt)
	}
	if !strings.Contains(prompt, "do not call dates before or equal to the current date") {
		t.Fatalf("writer prompt missing future-date guard: %s", prompt)
	}
}

func TestAnswerEmitsSearchEvents(t *testing.T) {
	searcher := &fakeSearchProvider{}
	base := &captureModel{}
	var events []SearchEvent
	ch, err := Answer(context.Background(), Request{
		Model:          base,
		SearchProvider: searcher,
		Mode:           ModeSpeed,
		Messages: []model.Message{
			model.NewUserMessage("what is vane"),
		},
		OnSearchEvent: func(_ context.Context, ev SearchEvent) {
			events = append(events, ev)
		},
	})
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	for range ch {
	}
	if !hasEventType(events, SearchEventClassification) ||
		!hasEventType(events, SearchEventStart) ||
		!hasEventType(events, SearchEventResearchStart) ||
		!hasEventType(events, SearchEventToolCall) ||
		!hasEventType(events, SearchEventToolResult) ||
		!hasEventType(events, SearchEventSourceBlock) ||
		!hasEventType(events, SearchEventWriterStart) ||
		!hasEventType(events, SearchEventWriterEnd) {
		t.Fatalf("events missing full search pipeline events: %#v", events)
	}
	end := lastEventOfType(events, SearchEventEnd)
	if end == nil {
		t.Fatalf("missing end event: %#v", events)
	}
	if end.ResultCount != 2 {
		t.Fatalf("end result count = %d, want 2 deduped results", end.ResultCount)
	}
	if len(end.Results) != 2 || end.Results[0].URL != "https://example.com/a#x" {
		t.Fatalf("end results = %#v, want deduped sources", end.Results)
	}
}

func hasEventType(events []SearchEvent, typ SearchEventType) bool {
	return lastEventOfType(events, typ) != nil
}

func lastEventOfType(events []SearchEvent, typ SearchEventType) *SearchEvent {
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Type == typ {
			return &events[i]
		}
	}
	return nil
}

func TestNormalizeMode(t *testing.T) {
	if got := NormalizeMode("quality"); got != ModeQuality {
		t.Fatalf("NormalizeMode quality = %s", got)
	}
	if got := NormalizeMode("weird"); got != ModeBalanced {
		t.Fatalf("NormalizeMode weird = %s, want balanced", got)
	}
}

func TestModeControlsSearchBreadth(t *testing.T) {
	tests := []struct {
		mode        Mode
		wantQueries int
		wantPerPage int
		wantTotal   int
	}{
		{mode: ModeSpeed, wantQueries: 1, wantPerPage: 8, wantTotal: 8},
		{mode: ModeBalanced, wantQueries: 3, wantPerPage: 8, wantTotal: 16},
		{mode: ModeQuality, wantQueries: 5, wantPerPage: 10, wantTotal: 24},
	}

	for _, tt := range tests {
		t.Run(string(tt.mode), func(t *testing.T) {
			if got := len(buildQueries("vane answering engine", tt.mode)); got != tt.wantQueries {
				t.Fatalf("query count = %d, want %d", got, tt.wantQueries)
			}
			if got := resultsPerQuery(tt.mode); got != tt.wantPerPage {
				t.Fatalf("resultsPerQuery = %d, want %d", got, tt.wantPerPage)
			}
			if got := totalResultLimit(tt.mode); got != tt.wantTotal {
				t.Fatalf("totalResultLimit = %d, want %d", got, tt.wantTotal)
			}
		})
	}
}

func TestModelInfoDoesNotMarshalAPIKey(t *testing.T) {
	payload, err := json.Marshal(ModelInfo{
		Name:    "cloud-model",
		BaseURL: "https://api.example.com/v1",
		APIKey:  "secret-key",
	})
	if err != nil {
		t.Fatalf("Marshal ModelInfo: %v", err)
	}
	if strings.Contains(string(payload), "secret-key") || strings.Contains(string(payload), "APIKey") {
		t.Fatalf("ModelInfo JSON leaked API key: %s", payload)
	}
	if !strings.Contains(string(payload), "cloud-model") {
		t.Fatalf("ModelInfo JSON missing non-sensitive model name: %s", payload)
	}
}
