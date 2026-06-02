package vane

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

func (a SearchAgent) planSearch(ctx context.Context, req SearchAgentRequest, classification Classification) Classification {
	fallback := fallbackSearchPlanForQuery(req.Query, classification, req.Mode, req.Sources, req.Now)
	plannerModel := firstModel(a.ResearchModel, a.ClassifierModel)
	if plannerModel == nil || strings.TrimSpace(req.Query) == "" {
		return applySearchPlanToClassification(classification, fallback)
	}
	planCtx, cancel := context.WithTimeout(ctx, plannerTimeout(req.Mode))
	defer cancel()
	ch, err := plannerModel.GenerateContent(planCtx, &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage(searchPlannerSystemPrompt()),
			model.NewUserMessage(searchPlannerUserPrompt(req, classification)),
		},
		GenerationConfig: model.GenerationConfig{Stream: false, Temperature: floatPtr(0), MaxTokens: intPtr(520)},
		ExtraFields:      req.ExtraFields,
	})
	if err != nil {
		return applySearchPlanToClassification(classification, fallback)
	}
	text, err := collectResponseText(planCtx, ch)
	if err != nil || strings.TrimSpace(text) == "" {
		return applySearchPlanToClassification(classification, fallback)
	}
	plan, err := parseSearchPlanJSON(text, req, classification)
	if err != nil {
		return applySearchPlanToClassification(classification, fallback)
	}
	plan = normalizeSearchPlan(plan, req.Query, classification, req.Mode, req.Sources, req.Now)
	if len(plan.Queries) == 0 {
		plan = fallback
	}
	return applySearchPlanToClassification(classification, plan)
}

func plannerTimeout(mode Mode) time.Duration {
	switch mode {
	case ModeSpeed:
		return 3 * time.Second
	case ModeQuality:
		return 10 * time.Second
	default:
		return 6 * time.Second
	}
}

func searchPlannerSystemPrompt() string {
	return `You are Vane's search query planner. Return compact JSON only.

Schema:
{
  "answer_goal": "what the final answer should accomplish",
  "topic": "clean research topic without command phrases",
  "language": "zh|en|other",
  "entities": ["key exact search entities, especially Latin names/acronyms/products/frameworks"],
  "report_sections": ["optional final-answer sections"],
  "queries": [
    {"query":"short search-engine keyword query","purpose":"why this query is useful","source":"web|academic|discussions|uploads","priority":1}
  ]
}

Rules:
- Separate the user's final answer goal from search queries.
- Search queries must be short keywords or focused sub-questions, not the user's full instruction.
- Remove task phrases such as "help me", "analyze", "generate a report", "write a report", "帮我", "分析一下", "生成", "写一份", "分析报告" from queries.
- Preserve entities, dates, locations, constraints, and the user's language.
- Put the main search entities in entities. For Chinese questions about Latin-named products, frameworks, APIs, libraries, models, or standards, keep those Latin entities exactly.
- If the user asks in Chinese, every query MUST be Chinese unless the user explicitly asks for English/global sources.
- For analysis/report requests, cover several evidence angles: background, latest developments, data/scale, official or primary sources, impact/risk, and differing views.
- Return only useful queries. Do not invent sources that are not enabled.`
}

func searchPlannerUserPrompt(req SearchAgentRequest, classification Classification) string {
	now := time.Now()
	if !req.Now.IsZero() {
		now = req.Now
	}
	return fmt.Sprintf("Current date: %s\nMode: %s\nEnabled sources: %s\nClassification intent: %s\nStandalone follow-up: %s\nConversation history:\n%s\n\nLatest user query:\n%s",
		now.Format("2006-01-02"),
		req.Mode,
		joinSources(normalizeSources(req.Sources, req.FileIDs)),
		classification.Intent,
		firstNonEmpty(classification.StandaloneFollowUp, req.Query),
		formatMessagesForPrompt(req.Messages),
		strings.TrimSpace(req.Query),
	)
}

type searchPlanJSON struct {
	AnswerGoal        string   `json:"answer_goal"`
	AnswerGoalAlt     string   `json:"answerGoal"`
	Topic             string   `json:"topic"`
	Language          string   `json:"language"`
	Entities          []string `json:"entities"`
	EntitiesAlt       []string `json:"search_entities"`
	EntitiesCamel     []string `json:"searchEntities"`
	ReportSections    []string `json:"report_sections"`
	ReportSectionsAlt []string `json:"reportSections"`
	Queries           []struct {
		Query    string `json:"query"`
		Purpose  string `json:"purpose"`
		Source   string `json:"source"`
		Priority int    `json:"priority"`
	} `json:"queries"`
}

func parseSearchPlanJSON(text string, req SearchAgentRequest, classification Classification) (SearchPlan, error) {
	raw := extractJSONObject(text)
	var parsed searchPlanJSON
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return SearchPlan{}, err
	}
	plan := SearchPlan{
		AnswerGoal: firstNonEmpty(parsed.AnswerGoal, parsed.AnswerGoalAlt),
		Topic:      parsed.Topic,
		Language:   parsed.Language,
	}
	plan.Entities = append(plan.Entities, parsed.Entities...)
	plan.Entities = append(plan.Entities, parsed.EntitiesAlt...)
	plan.Entities = append(plan.Entities, parsed.EntitiesCamel...)
	plan.ReportSections = append(plan.ReportSections, parsed.ReportSections...)
	plan.ReportSections = append(plan.ReportSections, parsed.ReportSectionsAlt...)
	for _, item := range parsed.Queries {
		source := SearchSource(strings.TrimSpace(item.Source))
		if source == "" {
			source = SearchSourceWeb
		}
		plan.Queries = append(plan.Queries, PlannedSearchQuery{
			Query:    item.Query,
			Purpose:  item.Purpose,
			Source:   source,
			Priority: item.Priority,
		})
	}
	if plan.AnswerGoal == "" && plan.Topic == "" && len(plan.Queries) == 0 {
		return SearchPlan{}, fmt.Errorf("vane: empty search plan")
	}
	return normalizeSearchPlan(plan, req.Query, classification, req.Mode, req.Sources, req.Now), nil
}

func applySearchPlanToClassification(classification Classification, plan SearchPlan) Classification {
	plan.AnswerGoal = strings.TrimSpace(firstNonEmpty(plan.AnswerGoal, classification.StandaloneFollowUp))
	classification.AnswerGoal = plan.AnswerGoal
	classification.SearchPlan = &plan
	if strings.TrimSpace(classification.StandaloneFollowUp) == "" {
		classification.StandaloneFollowUp = firstNonEmpty(plan.Topic, plan.AnswerGoal)
	}
	return classification
}

func normalizeSearchPlan(plan SearchPlan, rawQuery string, classification Classification, mode Mode, sources []SearchSource, now time.Time) SearchPlan {
	allowedSources := normalizeSources(append(append([]SearchSource{}, sources...), classification.Sources...), nil)
	if len(allowedSources) == 0 {
		allowedSources = classification.Sources
	}
	if len(allowedSources) == 0 {
		allowedSources = []SearchSource{SearchSourceWeb}
	}
	plan.AnswerGoal = strings.TrimSpace(firstNonEmpty(plan.AnswerGoal, classification.StandaloneFollowUp, rawQuery))
	plan.Topic = cleanSearchTaskQuery(firstNonEmpty(plan.Topic, classification.StandaloneFollowUp, rawQuery), now)
	if plan.Language == "" {
		if containsCJK(rawQuery) {
			plan.Language = "zh"
		} else {
			plan.Language = "en"
		}
	}
	entityContext := strings.Join(append([]string{
		rawQuery,
		plan.Topic,
		plan.AnswerGoal,
	}, searchPlanQueries(&plan)...), "\n")
	plan.Entities = normalizeSearchEntities(append(plan.Entities, sourcePlanEntities(entityContext)...))
	limit := plannedQueryLimit(mode)
	var out []PlannedSearchQuery
	seen := map[string]bool{}
	for _, item := range plan.Queries {
		query := cleanSearchTaskQuery(item.Query, now)
		if query == "" {
			continue
		}
		if looksLikeDegenerateSearchQuery(query) {
			continue
		}
		if containsCJK(rawQuery) && !queryLanguageExplicitlyAllowsEnglish(rawQuery) && !containsCJK(query) && !allowsEnglishTechnicalPlanQuery(rawQuery, plan, query) {
			continue
		}
		source := item.Source
		if source == "" || !hasSource(allowedSources, source) {
			source = SearchSourceWeb
		}
		key := string(source) + "\x00" + strings.ToLower(query)
		if seen[key] {
			continue
		}
		seen[key] = true
		item.Query = query
		item.Source = source
		if item.Priority <= 0 {
			item.Priority = len(out) + 1
		}
		out = append(out, item)
		if len(out) >= limit {
			break
		}
	}
	out = prioritizeTechnicalPlanQueries(rawQuery, plan, out, limit, allowedSources, seen)
	plan.Queries = out
	if len(plan.ReportSections) == 0 && looksLikeReportGoal(plan.AnswerGoal+" "+rawQuery) {
		plan.ReportSections = defaultReportSections(rawQuery)
	}
	return plan
}

func allowsEnglishTechnicalPlanQuery(rawQuery string, plan SearchPlan, query string) bool {
	contextText := strings.Join(append([]string{
		rawQuery,
		plan.Topic,
		plan.AnswerGoal,
	}, searchPlanQueries(&plan)...), "\n")
	return allowsEnglishSourceQueryForEntities(plan.Entities, contextText, query)
}

func prioritizeTechnicalPlanQueries(rawQuery string, plan SearchPlan, queries []PlannedSearchQuery, limit int, allowedSources []SearchSource, seen map[string]bool) []PlannedSearchQuery {
	overrides := technicalPlanQueryOverrides(rawQuery, plan)
	if len(overrides) == 0 {
		return queries
	}
	capacity := len(overrides) + len(queries)
	if limit > 0 && capacity > limit {
		capacity = limit
	}
	out := make([]PlannedSearchQuery, 0, capacity)
	for _, item := range overrides {
		source := item.Source
		if source == "" || !hasSource(allowedSources, source) {
			source = SearchSourceWeb
		}
		query := strings.TrimSpace(item.Query)
		if query == "" {
			continue
		}
		key := string(source) + "\x00" + strings.ToLower(query)
		if seen[key] {
			continue
		}
		seen[key] = true
		item.Query = query
		item.Source = source
		if item.Priority <= 0 {
			item.Priority = len(out) + 1
		}
		out = append(out, item)
		if len(out) >= limit {
			return out
		}
	}
	for _, item := range queries {
		out = append(out, item)
		if len(out) >= limit {
			return out
		}
	}
	return out
}

func technicalPlanQueryOverrides(rawQuery string, plan SearchPlan) []PlannedSearchQuery {
	context := strings.Join(append([]string{
		rawQuery,
		plan.Topic,
		plan.AnswerGoal,
	}, searchPlanQueries(&plan)...), "\n")
	if !needsOfficialSourcePlan(context) {
		return nil
	}
	var queries []string
	for _, entity := range firstNonEmptySearchEntities(plan.Entities, sourcePlanEntities(context)) {
		queries = append(queries,
			entity+" official documentation",
			entity+" API reference",
			entity+" GitHub repository",
		)
		if len(queries) >= 9 {
			break
		}
	}
	if len(queries) == 0 {
		return nil
	}
	queries = uniqueStrings(queries)
	out := make([]PlannedSearchQuery, 0, len(queries))
	for i, query := range queries {
		out = append(out, PlannedSearchQuery{
			Query:    query,
			Purpose:  "official technical source",
			Source:   SearchSourceWeb,
			Priority: i + 1,
		})
	}
	return out
}

func normalizeSearchEntities(entities []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, entity := range entities {
		entity = strings.TrimSpace(entity)
		entity = strings.Trim(entity, ".,;:()[]{}<>\"'")
		if entity == "" || containsCJK(entity) || isSourceQueryIntentTerm(strings.ToLower(entity)) {
			continue
		}
		key := strings.ToLower(entity)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, entity)
		if len(out) >= 8 {
			break
		}
	}
	return out
}

func firstNonEmptySearchEntities(primary []string, fallback []string) []string {
	if len(primary) > 0 {
		return primary
	}
	return fallback
}

func needsOfficialSourcePlan(context string) bool {
	if hasSourceQueryIntent(context) {
		return true
	}
	return containsAnyFold(context,
		"what is", "what are", "how does",
		"\u4ec0\u4e48\u662f", "\u4ecb\u7ecd", "\u5b9a\u4e49", "\u539f\u7406", "\u6280\u672f", "\u6846\u67b6",
	)
}

func sourcePlanEntities(context string) []string {
	seen := map[string]bool{}
	var out []string
	for _, token := range asciiQueryTokens(context) {
		token = strings.Trim(token, ".")
		if token == "" || !isLikelyEntityAnchor(token) || isSourceQueryIntentTerm(strings.ToLower(token)) {
			continue
		}
		key := strings.ToLower(token)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, token)
		if len(out) >= 3 {
			break
		}
	}
	return out
}

func fallbackSearchPlanForQuery(query string, classification Classification, mode Mode, sources []SearchSource, now time.Time) SearchPlan {
	topic := cleanSearchTaskQuery(firstNonEmpty(classification.StandaloneFollowUp, query), now)
	if topic == "" {
		topic = cleanSearchTaskQuery(query, now)
	}
	answerGoal := strings.TrimSpace(firstNonEmpty(classification.AnswerGoal, classification.StandaloneFollowUp, query))
	plan := SearchPlan{
		AnswerGoal: answerGoal,
		Topic:      topic,
		Language:   "en",
	}
	if containsCJK(query) {
		plan.Language = "zh"
	}
	for i, q := range fallbackPlannedQueries(topic, query, mode) {
		plan.Queries = append(plan.Queries, PlannedSearchQuery{
			Query:    q,
			Purpose:  fallbackQueryPurpose(q, i),
			Source:   SearchSourceWeb,
			Priority: i + 1,
		})
	}
	plan.ReportSections = defaultReportSections(query)
	return normalizeSearchPlan(plan, query, classification, mode, sources, now)
}

func fallbackPlannedQueries(topic string, rawQuery string, mode Mode) []string {
	if topic == "" {
		return nil
	}
	var queries []string
	if containsCJK(rawQuery) {
		queries = []string{topic}
		if looksLikeIncidentQuery(rawQuery) {
			queries = append(queries, topic+" 官方通报", topic+" 事故 伤亡", topic+" 交通 停电 损失", topic+" 影响")
		} else if looksLikeIndustryReportQuery(rawQuery) {
			queries = append(queries, topic+" 市场规模", topic+" 竞争格局", topic+" 政策", topic+" 趋势", topic+" 风险")
		} else {
			queries = append(queries, topic+" 最新进展", topic+" 数据 评价", topic+" 影响 风险", topic+" 官方 一手来源", topic+" 不同观点")
		}
	} else {
		queries = []string{topic, topic + " latest developments", topic + " data analysis", topic + " impact risks", topic + " official sources", topic + " opposing views"}
	}
	queries = uniqueStrings(queries)
	limit := plannedQueryLimit(mode)
	if len(queries) > limit {
		return queries[:limit]
	}
	return queries
}

func plannedQueryLimit(mode Mode) int {
	switch mode {
	case ModeSpeed:
		return 3
	case ModeQuality:
		return 12
	default:
		return 6
	}
}

func cleanSearchTaskQuery(query string, now time.Time) string {
	query = strings.TrimSpace(resolveRelativeDateQuery(query, now))
	if query == "" {
		return ""
	}
	replacements := []string{
		"请帮我", "", "为我", "", "帮我", "", "帮忙", "", "请问", "", "请", "", "麻烦", "",
		"介绍一下", "", "介绍-下", "", "介绍下", "", "介绍", "", "讲一下", "", "说一下", "",
		"什么是", "", "什么叫", "", "是啥", "",
		"给我生成", "", "生成一份", "", "生成一个", "", "生成", "",
		"写一份", "", "写一个", "", "写篇", "",
		"分析一下", "", "分析下", "",
		"看一下", "", "看看", "", "搜索一下", "", "搜索", "", "查找", "",
		"的分析报告", "", "分析报告", "", "研究报告", "", "行业报告", "",
		"help me", "", "please", "", "generate", "", "write a", "", "write an", "",
		"analysis report", "", "report about", "",
	}
	replacer := strings.NewReplacer(replacements...)
	query = replacer.Replace(query)
	query = strings.Trim(query, " \t\r\n,，。.!！?？:：;；-—")
	for strings.Contains(query, "  ") {
		query = strings.ReplaceAll(query, "  ", " ")
	}
	return strings.TrimSpace(query)
}

func hasTaskLanguage(query string) bool {
	return containsAnyFold(query,
		"为我", "帮我", "请帮", "请问", "给我", "介绍一下", "介绍下", "什么是", "什么叫", "生成", "写一份", "写一个", "分析一下", "分析报告", "研究报告", "行业报告",
		"help me", "please", "generate", "write a", "analysis report",
	)
}

func looksLikeDegenerateSearchQuery(query string) bool {
	query = strings.TrimSpace(query)
	if query == "" {
		return true
	}
	runes := []rune(query)
	if len(runes) == 1 && containsCJK(query) {
		return true
	}
	switch strings.ToLower(query) {
	case "为", "我", "请", "帮", "搜", "查", "看", "下", "为我", "帮我", "请问", "介绍", "分析", "一下":
		return true
	default:
		return false
	}
}

func looksLikeReportGoal(query string) bool {
	return containsAnyFold(query, "报告", "分析", "report", "analysis")
}

func looksLikeIncidentQuery(query string) bool {
	return containsAnyFold(query, "事故", "伤亡", "灾害", "大风", "龙卷风", "暴雨", "强对流", "停电", "incident", "casualties", "storm")
}

func looksLikeIndustryReportQuery(query string) bool {
	return containsAnyFold(query, "行业", "市场", "产业", "竞争", "industry", "market", "sector")
}

func defaultReportSections(query string) []string {
	if !looksLikeReportGoal(query) {
		return nil
	}
	if containsCJK(query) {
		return []string{"背景", "关键事实", "数据与证据", "影响与风险", "结论"}
	}
	return []string{"Background", "Key facts", "Evidence", "Impact and risks", "Conclusion"}
}

func fallbackQueryPurpose(query string, index int) string {
	switch index {
	case 0:
		return "core topic"
	case 1:
		return "latest developments"
	case 2:
		return "data and evidence"
	case 3:
		return "impact and risk"
	default:
		return "additional perspective"
	}
}

func searchPlanQueries(plan *SearchPlan) []string {
	if plan == nil {
		return nil
	}
	out := make([]string, 0, len(plan.Queries))
	for _, item := range plan.Queries {
		if strings.TrimSpace(item.Query) != "" {
			out = append(out, strings.TrimSpace(item.Query))
		}
	}
	return out
}

func plannedResearchQueries(req ResearchRequest, iterations int) []string {
	queries := searchPlanQueries(req.SearchPlan)
	if len(queries) == 0 {
		queries = searchPlanQueries(req.Classification.SearchPlan)
	}
	if len(queries) == 0 {
		fallback := fallbackSearchPlanForQuery(req.Query, req.Classification, req.Mode, req.Sources, req.Now)
		queries = searchPlanQueries(&fallback)
	}
	if len(queries) == 0 {
		queries = buildResearchQueries(req.Query, req.Mode, iterations, req.Now)
	}
	queries = cleanTaskQueries(queries, req.Now)
	queries = uniqueStrings(queries)
	if iterations > 0 && len(queries) > iterations {
		return queries[:iterations]
	}
	return queries
}

func plannedQueriesForSource(plan *SearchPlan, source SearchSource) []string {
	if plan == nil {
		return nil
	}
	var out []string
	for _, item := range plan.Queries {
		if strings.TrimSpace(item.Query) == "" {
			continue
		}
		if item.Source == "" || item.Source == source || source == SearchSourceWeb {
			out = append(out, item.Query)
		}
	}
	return uniqueStrings(out)
}

func shouldReplaceWithPlannedQueries(plan *SearchPlan, source SearchSource, queries []string) bool {
	if len(plannedQueriesForSource(plan, source)) == 0 {
		return false
	}
	for _, query := range queries {
		if usefulTechnicalToolQuery(plan, query) {
			continue
		}
		if looksLikeDegenerateSearchQuery(query) || hasTaskLanguage(query) || looksLikeVerboseSearchQuery(query) {
			return true
		}
	}
	return false
}

func usefulTechnicalToolQuery(plan *SearchPlan, query string) bool {
	if plan == nil {
		return false
	}
	return allowsEnglishTechnicalPlanQuery("", *plan, query)
}

func cleanTaskQueries(queries []string, now time.Time) []string {
	out := make([]string, 0, len(queries))
	for _, query := range queries {
		cleaned := cleanSearchTaskQuery(query, now)
		if cleaned != "" {
			out = append(out, cleaned)
		}
	}
	return uniqueStrings(out)
}

func formatSearchPlanForResearch(plan *SearchPlan) string {
	if plan == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("<query_plan>\n")
	if strings.TrimSpace(plan.AnswerGoal) != "" {
		fmt.Fprintf(&b, "<answer_goal>%s</answer_goal>\n", xmlishEscape(plan.AnswerGoal))
	}
	if strings.TrimSpace(plan.Topic) != "" {
		fmt.Fprintf(&b, "<topic>%s</topic>\n", xmlishEscape(plan.Topic))
	}
	if strings.TrimSpace(plan.Language) != "" {
		fmt.Fprintf(&b, "<language>%s</language>\n", xmlishEscape(plan.Language))
	}
	if len(plan.Entities) > 0 {
		b.WriteString("<entities>")
		for _, entity := range plan.Entities {
			if strings.TrimSpace(entity) != "" {
				fmt.Fprintf(&b, "<entity>%s</entity>", xmlishEscape(entity))
			}
		}
		b.WriteString("</entities>\n")
	}
	for _, item := range plan.Queries {
		if strings.TrimSpace(item.Query) == "" {
			continue
		}
		fmt.Fprintf(&b, `<query source="%s" priority="%d" purpose="%s">%s</query>`+"\n", xmlishEscape(string(item.Source)), item.Priority, xmlishEscape(item.Purpose), xmlishEscape(item.Query))
	}
	if len(plan.ReportSections) > 0 {
		b.WriteString("<report_sections>")
		for _, section := range plan.ReportSections {
			if strings.TrimSpace(section) != "" {
				fmt.Fprintf(&b, "<section>%s</section>", xmlishEscape(section))
			}
		}
		b.WriteString("</report_sections>\n")
	}
	b.WriteString("</query_plan>")
	return b.String()
}
