package vane

import (
	"context"
	"encoding/json"
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
	if len(searcher.queries) != 3 {
		t.Fatalf("queries = %v, want 3 balanced queries", searcher.queries)
	}
	if base.req == nil || len(base.req.Messages) == 0 {
		t.Fatal("base model did not receive writer request")
	}
	prompt := base.req.Messages[0].Content
	if !strings.Contains(prompt, "Alpha fact") || !strings.Contains(prompt, "[number]") {
		t.Fatalf("writer prompt missing source context/citation instruction: %s", prompt)
	}
	if strings.Count(prompt, `https://example.com/a`) != 1 {
		t.Fatalf("duplicate URL was not deduplicated in prompt: %s", prompt)
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
	if len(events) != 4 {
		t.Fatalf("events = %#v, want start/query/results/end", events)
	}
	if events[0].Type != SearchEventStart || events[1].Type != SearchEventQuery || events[2].Type != SearchEventResults || events[3].Type != SearchEventEnd {
		t.Fatalf("event order = %#v", []SearchEventType{events[0].Type, events[1].Type, events[2].Type, events[3].Type})
	}
	if events[3].ResultCount != 2 {
		t.Fatalf("end result count = %d, want 2 deduped results", events[3].ResultCount)
	}
	if len(events[3].Results) != 2 || events[3].Results[0].URL != "https://example.com/a#x" {
		t.Fatalf("end results = %#v, want deduped sources", events[3].Results)
	}
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
