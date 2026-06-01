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
	Sources            []SearchSource
	FileIDs            []string
	EmbeddingProvider  EmbeddingProvider
	WidgetProviders    WidgetProviders
	MaxIterations      int
	OnSearchError      func(error)
	OnSearchEvent      func(context.Context, SearchEvent)
	Now                time.Time
}

type SearchOptions struct {
	Page   int
	Limit  int
	Source SearchSource
}

type SearchResult struct {
	Title   string       `json:"title"`
	URL     string       `json:"url"`
	Content string       `json:"content"`
	Source  SearchSource `json:"source,omitempty"`
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
	SearchEventStart          SearchEventType = "start"
	SearchEventQuery          SearchEventType = "query"
	SearchEventResults        SearchEventType = "results"
	SearchEventEnd            SearchEventType = "end"
	SearchEventError          SearchEventType = "error"
	SearchEventClassification SearchEventType = "classification"
	SearchEventResearchStart  SearchEventType = "research_start"
	SearchEventResearchStep   SearchEventType = "research_step"
	SearchEventToolCall       SearchEventType = "tool_call"
	SearchEventToolResult     SearchEventType = "tool_result"
	SearchEventWidget         SearchEventType = "widget"
	SearchEventSourceBlock    SearchEventType = "source_block"
	SearchEventWriterStart    SearchEventType = "writer_start"
	SearchEventWriterDelta    SearchEventType = "writer_delta"
	SearchEventWriterEnd      SearchEventType = "writer_end"
)

type SearchEvent struct {
	Type           SearchEventType  `json:"type"`
	Mode           Mode             `json:"mode,omitempty"`
	Query          string           `json:"query,omitempty"`
	Queries        []string         `json:"queries,omitempty"`
	QueryIndex     int              `json:"query_index,omitempty"`
	QueryTotal     int              `json:"query_total,omitempty"`
	ResultCount    int              `json:"result_count,omitempty"`
	Results        []SearchResult   `json:"results,omitempty"`
	Error          string           `json:"error,omitempty"`
	Classification *Classification  `json:"classification,omitempty"`
	Step           int              `json:"step,omitempty"`
	StepTotal      int              `json:"step_total,omitempty"`
	Action         string           `json:"action,omitempty"`
	ToolCall       *ToolCallEvent   `json:"tool_call,omitempty"`
	ToolResult     *ToolResultEvent `json:"tool_result,omitempty"`
	Widget         *WidgetResult    `json:"widget,omitempty"`
	SourceBlock    *SourceBlock     `json:"source_block,omitempty"`
	Message        string           `json:"message,omitempty"`
	Metadata       map[string]any   `json:"metadata,omitempty"`
}

type SearchProvider interface {
	Search(ctx context.Context, query string, opts SearchOptions) ([]SearchResult, error)
}

type AnsweringModel struct {
	Base               model.Model
	ModelInfo          ModelInfo
	Mode               Mode
	SearchProvider     SearchProvider
	Sources            []SearchSource
	FileIDs            []string
	EmbeddingProvider  EmbeddingProvider
	WidgetProviders    WidgetProviders
	MaxIterations      int
	SystemInstructions string
	OnSearchError      func(error)
	OnSearchEvent      func(context.Context, SearchEvent)
}

type SearchSource string

const (
	SearchSourceWeb         SearchSource = "web"
	SearchSourceDiscussions SearchSource = "discussions"
	SearchSourceAcademic    SearchSource = "academic"
	SearchSourceUploads     SearchSource = "uploads"
)

type Classification struct {
	ShouldSearch bool           `json:"should_search"`
	Intent       string         `json:"intent,omitempty"`
	Reason       string         `json:"reason,omitempty"`
	Sources      []SearchSource `json:"sources,omitempty"`
	NeedWeather  bool           `json:"need_weather,omitempty"`
	NeedStock    bool           `json:"need_stock,omitempty"`
	NeedCalc     bool           `json:"need_calc,omitempty"`
	NeedUploads  bool           `json:"need_uploads,omitempty"`
	SkipReason   string         `json:"skip_reason,omitempty"`
}

type ToolCallEvent struct {
	ID     string         `json:"id,omitempty"`
	Name   string         `json:"name"`
	Query  string         `json:"query,omitempty"`
	Args   map[string]any `json:"args,omitempty"`
	Step   int            `json:"step,omitempty"`
	Source SearchSource   `json:"source,omitempty"`
}

type ToolResultEvent struct {
	ID          string         `json:"id,omitempty"`
	Name        string         `json:"name"`
	Query       string         `json:"query,omitempty"`
	ResultCount int            `json:"result_count,omitempty"`
	Results     []SearchResult `json:"results,omitempty"`
	Error       string         `json:"error,omitempty"`
	Step        int            `json:"step,omitempty"`
	Source      SearchSource   `json:"source,omitempty"`
}

type SourceBlock struct {
	Results []SearchResult `json:"results,omitempty"`
	Widgets []WidgetResult `json:"widgets,omitempty"`
}

type WidgetKind string

const (
	WidgetWeather WidgetKind = "weather"
	WidgetStock   WidgetKind = "stock"
	WidgetCalc    WidgetKind = "calculation"
)

type WidgetResult struct {
	Kind    WidgetKind     `json:"kind"`
	Query   string         `json:"query,omitempty"`
	Title   string         `json:"title,omitempty"`
	Content string         `json:"content,omitempty"`
	Error   string         `json:"error,omitempty"`
	Data    map[string]any `json:"data,omitempty"`
}

type WidgetProviders struct {
	Weather     WidgetProvider
	Stock       WidgetProvider
	Calculation WidgetProvider
}

type WidgetProvider interface {
	RunWidget(ctx context.Context, query string) (WidgetResult, error)
}

type EmbeddingProvider interface {
	SearchUploads(ctx context.Context, query string, fileIDs []string, limit int) ([]SearchResult, error)
}
