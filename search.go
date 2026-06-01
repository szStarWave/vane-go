package vane

import (
	"context"
	"errors"
	"strings"
	"time"

	websurfx "github.com/szStarWave/websurfx-go"
)

type WebsurfxProvider struct {
	client *websurfx.Client
}

func NewWebsurfxProvider(client *websurfx.Client) (*WebsurfxProvider, error) {
	if client == nil {
		var err error
		client, err = websurfx.New(websurfx.Options{
			Timeout:  10 * time.Second,
			CacheTTL: 5 * time.Minute,
		})
		if err != nil {
			return nil, err
		}
	}
	return &WebsurfxProvider{client: client}, nil
}

func MustNewWebsurfxProvider() *WebsurfxProvider {
	provider, err := NewWebsurfxProvider(nil)
	if err != nil {
		panic(err)
	}
	return provider
}

func (p *WebsurfxProvider) Search(ctx context.Context, query string, opts SearchOptions) ([]SearchResult, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("vane: websurfx provider is nil")
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}
	page := opts.Page
	if page < 1 {
		page = 1
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	response := p.client.Search(ctx, websurfx.Query{Text: query, Page: page})
	results := make([]SearchResult, 0, min(limit, len(response.Results)))
	for _, item := range response.Results {
		title := strings.TrimSpace(item.Title)
		url := strings.TrimSpace(item.URL)
		content := strings.TrimSpace(item.Description)
		if title == "" && content == "" {
			continue
		}
		results = append(results, SearchResult{
			Title:   firstNonEmpty(title, url),
			URL:     url,
			Content: firstNonEmpty(content, title),
		})
		if len(results) >= limit {
			break
		}
	}
	if len(results) == 0 && len(response.Errors) > 0 {
		return nil, errors.New(response.Errors[0].Message)
	}
	return results, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func firstMap(values ...map[string]any) map[string]any {
	for _, value := range values {
		if len(value) > 0 {
			return value
		}
	}
	return nil
}

func mergeExtraFields(values ...map[string]any) map[string]any {
	var out map[string]any
	for _, value := range values {
		if len(value) == 0 {
			continue
		}
		if out == nil {
			out = map[string]any{}
		}
		for key, item := range value {
			out[key] = item
		}
	}
	return out
}
