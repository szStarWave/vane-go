package vane

import (
	"context"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

func (m *AnsweringModel) Info() model.Info {
	if m == nil || m.Base == nil {
		return model.Info{Name: "vane"}
	}
	info := m.Base.Info()
	if info.Name == "" {
		info.Name = "vane"
	}
	return info
}

func (m *AnsweringModel) GenerateContent(ctx context.Context, request *model.Request) (<-chan *model.Response, error) {
	if m == nil || m.Base == nil {
		return nil, modelErr("vane: base model is nil")
	}
	if request == nil {
		return nil, modelErr("vane: request is nil")
	}
	mode := m.Mode
	if mode == "" {
		mode = ModeBalanced
	}
	return Answer(ctx, Request{
		Model:              m.Base,
		ModelInfo:          m.ModelInfo,
		Messages:           request.Messages,
		GenerationConfig:   request.GenerationConfig,
		ExtraFields:        request.ExtraFields,
		Mode:               mode,
		SystemInstructions: m.SystemInstructions,
		SearchProvider:     m.SearchProvider,
		Sources:            m.Sources,
		FileIDs:            m.FileIDs,
		EmbeddingProvider:  m.EmbeddingProvider,
		WidgetProviders:    m.WidgetProviders,
		MaxIterations:      m.MaxIterations,
		OnSearchError:      m.OnSearchError,
		OnSearchEvent:      m.OnSearchEvent,
	})
}

type modelErr string

func (e modelErr) Error() string { return string(e) }
