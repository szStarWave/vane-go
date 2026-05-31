package vane

import (
	"context"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

type Mode string

const (
	ModeSpeed    Mode = "speed"
	ModeBalanced Mode = "balanced"
	ModeQuality  Mode = "quality"
)

func NormalizeMode(value string) Mode {
	switch Mode(strings.ToLower(strings.TrimSpace(value))) {
	case ModeSpeed:
		return ModeSpeed
	case ModeQuality:
		return ModeQuality
	default:
		return ModeBalanced
	}
}

type Request struct {
	Model              model.Model
	ModelInfo          ModelInfo
	Messages           []model.Message
	GenerationConfig   model.GenerationConfig
	ExtraFields        map[string]any
	Mode               Mode
	SystemInstructions string
	SearchProvider     SearchProvider
	OnSearchError      func(error)
	OnSearchEvent      func(context.Context, SearchEvent)
	Now                time.Time
}

type SearchOptions struct {
	Page  int
	Limit int
}

type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

type ModelInfo struct {
	Name               string `json:"name,omitempty"`
	ModelID            string `json:"model_id,omitempty"`
	Provider           string `json:"provider,omitempty"`
	BaseURL            string `json:"base_url,omitempty"`
	APIKey             string `json:"-"`
	SupportsImageInput bool   `json:"supports_image_input,omitempty"`
}

type SearchEventType string

const (
	SearchEventStart   SearchEventType = "start"
	SearchEventQuery   SearchEventType = "query"
	SearchEventResults SearchEventType = "results"
	SearchEventEnd     SearchEventType = "end"
	SearchEventError   SearchEventType = "error"
)

type SearchEvent struct {
	Type        SearchEventType `json:"type"`
	Mode        Mode            `json:"mode,omitempty"`
	Query       string          `json:"query,omitempty"`
	Queries     []string        `json:"queries,omitempty"`
	QueryIndex  int             `json:"query_index,omitempty"`
	QueryTotal  int             `json:"query_total,omitempty"`
	ResultCount int             `json:"result_count,omitempty"`
	Results     []SearchResult  `json:"results,omitempty"`
	Error       string          `json:"error,omitempty"`
}

type SearchProvider interface {
	Search(ctx context.Context, query string, opts SearchOptions) ([]SearchResult, error)
}

type AnsweringModel struct {
	Base               model.Model
	ModelInfo          ModelInfo
	Mode               Mode
	SearchProvider     SearchProvider
	SystemInstructions string
	OnSearchError      func(error)
	OnSearchEvent      func(context.Context, SearchEvent)
}
