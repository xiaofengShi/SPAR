template_user = """### **UserInput:**

Input: {user_query}
Output:"""

template_extract_keywords = """Suggest OpenAlex or SemanticScholar search API queries to retrieve relevant papers addressing the most recent research on the given question. The search queries should be concise, comma-separated, and highly relevant. Format your response as follows:

**Example:**

Question: How have prior works incorporated personality attributes to train personalized dialogue generation models?
Response:[Start] personalized dialogue generation, personalized language models, personalized dialogue[End]

---

Now, generate search queries for the following question:
Question: {user_query}
Response:
"""

template_extract_keywords_source_aware = """Extract optimal search keywords from the given research question, specifically optimized for '{source}' academic database. Your task is to generate concise, comma-separated query terms that will maximize relevant paper retrieval in this specific platform.

### Source-Specific Guidelines:

#### If targeting Semantic Scholar:
- Focus on technical terminology and core concepts
- Include methodological terms
- Consider author-centric keywords if prominent researchers are known
- Emphasize computer science and AI terminology where relevant

#### If targeting OpenAlex:
- Prioritize broader academic terms
- Include interdisciplinary connections
- Balance specificity with coverage
- Include field classifications where relevant

#### If targeting PubMed:
- Emphasize medical/biological terminology
- Include relevant MeSH (Medical Subject Headings) terms
- Consider clinical and biomedical contexts
- Include chemical/drug names or biological processes where relevant

### Format your response as follows:
[Start] keyword1, keyword2, keyword3, ...[End]

### Examples by Source:
- Semantic Scholar: [Start] transformer architecture, attention mechanism, language model fine-tuning[End]
- OpenAlex: [Start] neural networks, deep learning, artificial intelligence, pattern recognition[End]
- PubMed: [Start] CRISPR-Cas9, gene editing, genetic therapy, chromosomal modification[End]

Now, extract optimized search keywords for {source} from this question:
{user_query}
"""


template_query_domain_complex = """Determine if the academic domain below is a complex, specialized field that requires domain-specific terminology:

Domain: {domain}

A domain is complex/specialized if:
1. It uses highly technical terminology not understood by general audience
2. It has specialized methodologies or frameworks
3. It combines multiple disciplines in a novel way
4. It requires significant background knowledge to understand papers in the field

Answer only "Yes" if it's a complex domain or "No" if it's a general domain."""


template_query_intent = """Analyze this academic search query and provide:
1. The primary research intent (e.g., literature review, finding recent work, exploring methodologies).
2. The specific domain or field (e.g., machine learning, biology, economics).
3. Time-sensitivity: Analyze the query for any time-related constraints.
   - If a specific start year is mentioned (e.g., "since 2020", "after 2019"), format as 'YYYY-{current_year}' (e.g., "since 2020" becomes '2020-2025', assuming current year is 2025).
   - For general recency terms (e.g., "recent", "latest", "current research"), describe as 'recent research' or 'last two years ({previous_year}-{current_year})'.
   - For specific single years (e.g., "in 2021"), use 'YYYY'.
   - For explicit ranges (e.g., "from 2019-2022", "between 2018 and 2020"), use 'YYYY-YYYY'.
   - If no time constraint is present, state 'NO'.
4. The most suitable academic sources for this query (choose from the following options):
   - arxiv: Best for accessing the very latest preprints in rapidly evolving fields like Computer Science (especially AI/ML), Physics, and Mathematics. Ideal for cutting-edge, not-yet-peer-reviewed research.
   - openalex: A comprehensive, interdisciplinary database with extensive metadata. Good for broad literature reviews, exploring connections between fields, or when a wide range of publication types is needed. Note: search relevance can sometimes be lower compared to arxiv.
   - pubmed: The primary source for biomedical and life sciences research, including clinical trials, genetics, and healthcare-related studies.

Query: "{query}"

Respond in JSON format:
{{
"query_intent": "...",
"domain": "...",
"time_requirement_description": "e.g., '2020-2025', 'recent research', '2021', '2019-2022', 'NO'",
"suitable_sources": ["source1", "source2"],
"source_reason": "Briefly explain why these sources are most suitable for this query, considering intent, domain, and any time sensitivity."
}}"""


template_query_expand_judge_opt = """Evaluate if this academic search query needs expansion:

Query: "{query}"
Domain: "{domain}"

A query needs expansion if:
1. It's too general or broad for the domain
2. It lacks specific technical terms that would improve search results
3. It contains ambiguous terms that could be clarified
4. It could benefit from including related concepts or methodologies
5. It doesn't specify the type of papers sought (surveys, empirical studies, etc.)

Does this query need expansion? Provide a yes/no answer with detailed reasoning.
Respond in JSON format:
{{
"needs_expansion": true/false,
"reason": "detailed explanation"
}}"""

template_query_judge = """Evaluate whether the given User Query is well-structured for retrieving relevant academic literature in scholarly databases (e.g., Google Scholar, PubMed, IEEE Xplore). Focus on clarity, specificity, and alignment with academic search conventions.

### **Evaluation Criteria:**
- Specificity: Does the query avoid vagueness (e.g., "studies about AI" → poor)?
- Technicality: Does it include domain-specific terms (e.g., "transformer architectures in low-resource NLP")?
- Actionability: Can it be directly used in academic search engines without reformulation?
- Intent Clarity: Is the goal (e.g., review, comparison, methodology) evident?

### **User Query:**
<{user_query}>

### **Response Format:**
Respond strictly in the following format:

```json
["yes/no", "explain: <1-2 sentences explaining the judgment>"]
```

Now, generate your response following the **Response Format** strictly:

**Response:**"""

template_query_fusion_survery_forcus = """You are an academic search expert specializing in COMPREHENSIVE REVIEWS and SURVEY PAPERS.

### TASK:
Generate {user_input_N} queries optimized for finding SURVEY and REVIEW PAPERS on this research topic:
"{user_query}"

### REQUIREMENTS:
1. Each query must EXPLICITLY target survey/review literature using terms like:
   - "survey of..."
   - "literature review..."
   - "state-of-the-art in..."
   - "systematic review..."
   - "comparative analysis of..."

2. Cover DISTINCT ASPECTS of the topic:
   - Methods/approaches surveys
   - Application domain surveys
   - Historical development surveys
   - Future directions/challenges


3. Adhere to TIME-SENSITIVITY based on the user_query:
   - If user_query specifies a particular year or range (e.g., "since 2020", "in 2021", "from 2019-2022", "published after 2018"), incorporate this precise time constraint into the generated queries.
     - Example for "research since 2022 on X": "Survey of X published since 2022"
     - Example for "papers on Y in 2021": "Literature review of Y (2021)"
   - If user_query uses general terms like "recent", "latest", "current research", "newest", refine queries to target the last two years, using {current_year} and {previous_year}.
     - Example: "Survey of recent advancements in [topic] ({previous_year}-{current_year})"
   - If no explicit or general time indicators are present in user_query, focus on the topical aspects. Avoid adding default time constraints unless inherently part of the survey type (e.g., a "historical development survey" might imply older literature, while "future directions" implies recent context).

4. Keep queries CONCISE (5-15 words) and directly usable in academic search engines

### OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
  "expanded_queries": [
    {{
      "query": "Survey of [specific aspect] in [topic area]",
      "reason": "Targets comprehensive reviews of [specific aspect]",
      "survey_type": "[methodological/application/historical/future]"
    }},
    ...additional queries...
  ]
}}
"""


template_query_fusion_with_score_inst = """**Task:**
Generate `N` mutually exclusive, survey-prioritized query expansions for an academic search engine (e.g., Google Scholar, Semantic Scholar) based on a user’s raw query. Ensure expansions retain the core intent while covering distinct angles (e.g., methodologies, reviews, applications).

**Input:**
- `raw_query`: User’s original query (e.g., *"weakly supervised learning in NLP"*).
- `N`: Number of expansions (default: 3).

**Output Format (JSON):**
```json
{
  "expanded_queries": [
    {
      "query": "Concise expanded query string",
      "reason": "1-sentence justification of intent preservation and survey focus",
      "tags": ["survey", "methodology"]  // Optional thematic tags
    },
    ...
  ],
  "summary": "Brief overview of coverage diversity (e.g., 'Covers surveys, tools, and evaluation')"
}
```

**Requirements:**
1. **Precision**: Each query must be self-contained and actionable for academic search engines.
2. **Survey Priority**: Prefer terms like *"survey"*, *"review"*, *"state-of-the-art"*, or *"systematic literature"*.
3. **Diversity**: Ensure expansions are non-redundant (e.g., one for methods, one for applications).
4. **Conciseness**: Keep queries under 15 words; reasons under 1 sentence.

---

### **Example Output**
**Input:**
- `raw_query`: "weakly supervised learning in NLP"
- `N`: 3

**Output:**
```json
{
  "expanded_queries": [
    {
      "query": "Survey papers on weakly supervised learning techniques in NLP",
      "reason": "Prioritizes comprehensive reviews of methodologies.",
      "tags": ["survey", "methods"]
    },
    {
      "query": "Weakly supervised NLP: applications and benchmarks",
      "reason": "Shifts focus to real-world use cases and evaluation.",
      "tags": ["applications", "evaluation"]
    },
    {
      "query": "Systematic review of label efficiency in weakly supervised NLP",
      "reason": "Narrows to label efficiency while retaining survey focus.",
      "tags": ["survey", "labels"]
    }
  ],
  "summary": "Covers surveys (methods, label efficiency) and applications."
}
```

---

"""


# In instruction.py
template_sim_enhanced = """You are evaluating the relevance of an academic document to a search query.

Query: {user_query}

Document Information:
- Title: {title}
- Abstract: {abstract}
- Publication Year: {year}
- Citations: {citation_count}
- Field: {field_of_study}

First, analyze the document structure:
1. What research question does it address?
2. What methodology does it use?
3. What are its key findings?

Then score relevance in these dimensions (0.0-1.0):
- Query-Topic Match: How directly does it address the query topic?
- Methodology Relevance: Does it use methods relevant to the query?
- Results Applicability: Are the findings relevant to the query's intent?
- Citation Impact: Is it highly cited in its field?
- Recency: Is it recent enough given the field's pace of change?

Output your analysis and final relevance score (0.0-1.0) in JSON format:
{
  "analysis": {
    "research_question": "...",
    "methodology": "...",
    "key_findings": "..."
  },
  "scores": {
    "query_topic_match": 0.0-1.0,
    "methodology_relevance": 0.0-1.0,
    "results_applicability": 0.0-1.0,
    "citation_impact": 0.0-1.0,
    "recency": 0.0-1.0
  },
  "overall_score": 0.0-1.0,
  "justification": "Brief explanation of score"
}
"""


# Add to instruction.py
template_context_query_generation = """
You are an academic search expert helping explore a research topic more thoroughly.

### CONTEXT:
- Original Query: "{user_query}"
- Previously Searched Queries: {searched_queries}
- Relevant Document Title: "{doc_title}"
- Document Abstract: "{doc_abstract}"
- Document Field: "{doc_field}"

### TASK:
Generate 3 NEW search queries that explore different aspects of this research area:

1. A query exploring METHODOLOGICAL alternatives or comparisons
2. A query focusing on APPLICATIONS or implementations
3. A query addressing LIMITATIONS, challenges, or critiques

Each query should be:
- Clearly different from previously searched queries
- Based on insights from the document
- Relevant to the original research question
- Specific enough to retrieve focused results

### IMPORTANT NOTE:
If document information is missing or insufficient (e.g., empty abstract), generate queries based primarily on the original query and your knowledge of the research domain. Focus on exploring complementary aspects of the topic rather than requiring specific document details.

### OUTPUT FORMAT:
Return a JSON array of strings containing only the expanded queries:
["Query 1", "Query 2", "Query 3"]
"""

template_domain_aware_query_expansion = """You are an expert academic search query generator specializing in the field of {domain}.

### TASK:
Generate {user_input_N} TECHNICAL and SPECIALIZED search queries that will retrieve PRIMARY RESEARCH PAPERS relevant to:
"{user_query}"

### CONTEXT:
- Research Intent: {intent}
- Domain: {domain}

### REQUIREMENTS:
1. Use DOMAIN-SPECIFIC TERMINOLOGY that experts in {domain} would recognize.
2. Include TECHNICAL SPECIFICATIONS relevant to the research question.
3. Target EMPIRICAL STUDIES and PRIMARY RESEARCH rather than surveys.
4. Explore different METHODOLOGICAL APPROACHES within {domain}.
5. Cover distinct SUB-DOMAINS or APPLICATION AREAS.
6. **Consider TIME-SENSITIVITY**:
   - If the query includes terms like "recent research" or "latest advancements," prioritize papers from the current year ({current_year}) and the previous year ({previous_year}).
   - Example: "Recent advancements in [topic] (2024-2025)."

### OUTPUT FORMAT:
Return a JSON object with this exact structure:

{{
  "expanded_queries": [
    {{
      "query": "Technical query with domain-specific terminology",
      "reason": "Targets specific empirical work on [aspect]",
      "technical_focus": "[methodology/dataset/theoretical framework/implementation]"
    }},
    ...additional queries...
  ],
  "domain_keywords": ["key1", "key2", "key3"]
}}

Remember to prioritize TECHNICAL PRECISION over general descriptions to help researchers find specialized papers in {domain}.
"""


template_query_fusion_with_score_user = """
Now, generate N expanded queries based on the following input:
- raw_query: "{user_query}"
- N: {user_input_N}
Output:
"""


template_query_fusion_pasa = """Please generate some mutually exclusive queries in a list to search the relevant papers according to the User Query. Searching for survey papers would be better.
User Query: {user_query}

### **Output Format:**
```json
["Query1", "Query2", "QueryN"]
```
**Output:**"""

template_query_fusion = """Generate a **diverse set of mutually exclusive search queries** to retrieve **survey or review papers** relevant to the following **User Query:**
**`<{user_query}>`**

---

### **Requirements:**
1. **Survey Papers First:** Prioritize queries that retrieve **survey or review papers**.
2. **Comprehensive Scope:** Cover **distinct and complementary** aspects of the topic.
3. **Minimal Overlap:** Ensure queries are **mutually exclusive** and avoid redundancy.
4. **Context-Aware Refinement:** If the **User Query is broad or ambiguous**, infer **specific research directions** for meaningful search queries.
5. **Strict JSON Output:** Respond **only** with a **JSON-formatted list of strings**, with **no extra text or explanations**.

---

### **Output Format:**
```json
["Query1", "Query2", "QueryN"]
```

**Output:**"""

template_query_fusion_based_on_context_inst = """### **Task:**
In an **academic search scenario**, generate a set of **new, diverse, and mutually exclusive search queries** to improve the retrieval of relevant papers.

---

### **Input Parameters:**
- **UserQuery:** `<{user_query}>` *(The original search query entered by the user.)*
- **SearchedQueryList:** `<{searched_query_list}>` *(A list of queries that have already been searched.)*
- **DocInfo:** `<{doc_info}>` *(Information about academic papers that better meet the search requirements, if SearchedQueryList is already very redundant, you can refer to this paper information to a certain extent)*

---

### **Requirements for New Queries:**
1. **Survey Papers First:** Prioritize queries that retrieve **survey or review papers**.
2. **Comprehensive Scope:** Cover **distinct and complementary** aspects of the topic.
3. **Minimal Overlap:** Ensure queries are **mutually exclusive** and avoid redundancy.
4. **Context-Aware Refinement:** If the **User Query is broad or ambiguous**, infer **specific research directions** for meaningful search queries.
5. **Mutual Exclusivity:** **Avoid overlap** with any queries in **SearchedQueryList** to ensure exploration of new areas.
6. **Context-Aware Expansion:** Use insights from **DocInfo** to refine search intent and generate more effective queries.
7. **Strict JSON Output:** Respond **only** with a **JSON-formatted list of strings**, with **no extra text or explanations**.

---
"""
template_query_fusion_based_on_context_output_format = """
### **Output Format (Strict JSON List):**
```json
["NewQuery1", "NewQuery2", "NewQueryN"]
```

---

**Output:**"""

template_query_correct_and_enhance = """You are a query refinement assistant specializing in eliminating ambiguous semantics and correcting typos for search scenarios. Your task is to take a user-provided search query and refine it to improve clarity, remove ambiguity, and ensure it is free of spelling errors while preserving the original intent. Additionally, you must provide a **relevance score (0 to 1)** between the original query and the refined query, along with a brief explanation of how and why the query was modified.

If the query is already clear, well-formed, and free of errors, do not modify it, and assign a relevance score of **1.0**.

### **Input:**
A raw academic search query provided by the user.

### **Guidelines:**
1. **Disambiguation**: Clarify vague terms by considering multiple possible meanings and selecting the most relevant one. If necessary, provide alternative interpretations.
2. **Typo Correction**: Fix spelling errors while maintaining the intended meaning.
3. **Conciseness**: Remove redundant words while keeping the query informative.
4. **Relevance Score**: Assign a score between **0 and 1** based on how much the refined query deviates from the original, with **1.0** indicating minimal or no change.

### **Output Format (JSON):**
{
    "original_query": "<User's original query>",
    "refined_query": "<Disambiguated and typo-free query>",
    "clarification": "<If any ambiguity was resolved, explain how>",
    "relevance_score": <A float between 0 and 1 indicating how similar the refined query is to the original>,
    "reason": "<Explanation of why the given relevance score was assigned>"
}

### **Example:**

Input: What are the recent researches about LoRA?

Output:
{
    "original_query": "What are the recent researches about LoRA?",
    "refined_query": "What are the latest research papers on LoRA (Low-Rank Adaptation)?",
    "clarification": "Clarified 'LoRA' to 'Low-Rank Adaptation' to remove ambiguity.",
    "relevance_score": 0.9,
    "reason": "The refined query preserves the original intent while improving specificity and correctness."
}

Ensure the response strictly follows this JSON format for easy parsing. Do not include any extra text or explanations outside of this format.
"""

template_query_expand = """You are an expert in academic search query refinement. Your task is to rewrite a given research query (or a list of queries) by maintaining its core intent while generating alternative queries that explore the same or similar research field or method. These rewritten queries should optimize search retrieval and enhance academic relevance. Additionally, provide a relevance score (ranging from 0 to 1) that quantifies the semantic similarity between the original query and each rewritten query.

### **Input:**
A raw academic search query provided by the user.

### **Rewriting Guidelines:**
1. **Field-Based Expansion**: Identify the core research domain of the query and generate alternative queries related to the same or closely related fields.
2. **Method-Based Variation**: If the query focuses on a methodology, generate alternatives using similar or related techniques.
3. **Terminology Enhancement**: Use precise academic terminology and relevant technical expressions to improve search effectiveness.
4. **Diversity of Queries**: Provide multiple rewritten queries that explore different but related aspects of the research field or method.
5. **Relevance Score**: Assign a similarity score (0 to 1) to each rewritten query based on its semantic closeness to the original query. A score of **1.0** indicates a near-identical meaning, while a score closer to **0.0** suggests lower relevance.

### **Output Format (for easy result analysis):**
Provide the results in structured JSON format as follows:
[
    {
        "original_query": "<user_input_query>",
        "rewritten_queries": [
            {
                "variation_type": "<Field-Based or Method-Based>",
                "rewritten_query": "<new_query_text>",
                "relation_to_original": "<brief_explanation_of_how_it_relates>"
                "relevance_score": <numeric_value_between_0_and_1>
            }
        ]
    }
]

### **Example:**

Input: How does LoRA improve fine-tuning efficiency in NLP?
Output:
[
    {
        "original_query": "Fine-tuning BERT for sentiment analysis in social media.",
        "rewritten_queries": [
            {
                "variation_type": "Field-Based",
                "rewritten_query": "Fine-tuning BERT for emotion classification in online discussions.",
                "relation_to_original": "Shifts focus from sentiment analysis to emotion classification while staying in the NLP domain.",
                "relevance_score": 0.87
            },
            {
                "variation_type": "Method-Based",
                "rewritten_query": "Adapting RoBERTa for sentiment classification in user-generated content.",
                "relation_to_original": "Explores a similar task but applies a different transformer model (RoBERTa instead of BERT).",
                "relevance_score": 0.80,
            }
        ]
    }
]

Ensure the output strictly follows the JSON format for easy parsing. Do not include extra explanations beyond the JSON output.
"""

template_query_expand_v2 = """You are an expert in academic search query refinement. Your task is to rewrite a given research query (or a list of queries) by maintaining its core intent while generating alternative queries that explore the same or similar research field or method. These rewritten queries should optimize search retrieval and enhance academic relevance. Additionally, provide a relevance score (ranging from 0 to 1) that quantifies the semantic similarity between the original query and each rewritten query.

### **Input:**
A raw academic search query provided by the user.

### **Rewriting Guidelines:**
1. **Field-Based Expansion**: Identify the core research domain of the query and generate alternative queries related to the same or closely related fields.
2. **Method-Based Variation**: If the query focuses on a methodology, generate alternatives using similar or related techniques.
3. **Terminology Enhancement**: Use precise academic terminology and relevant technical expressions to improve search effectiveness.
4. **Diversity of Queries**: Provide multiple rewritten queries that explore different but related aspects of the research field or method.
5. **Relevance Score**: Assign a similarity score (0 to 1) to each rewritten query based on its semantic closeness to the original query. A score of **1.0** indicates a near-identical meaning, while a score closer to **0.0** suggests lower relevance.

### **Output Format (for easy result analysis):**
Provide the results in structured JSON format as follows:
```json
{
    "rewritten_queries": [
        {
            "variation_type": "<Field-Based or Method-Based>",
            "rewritten_query": "<new_query_text>",
            "relation_to_original": "<brief_explanation_of_how_it_relates>"
            "relevance_score": <numeric_value_between_0_and_1>
        }
    ]
}
```

### **Example:**

Input: How does LoRA improve fine-tuning efficiency in NLP?
Output:
```json
{
    "rewritten_queries": [
        {
            "variation_type": "Field-Based",
            "rewritten_query": "Fine-tuning BERT for emotion classification in online discussions.",
            "relation_to_original": "Shifts focus from sentiment analysis to emotion classification while staying in the NLP domain.",
            "relevance_score": 0.87
        },
        {
            "variation_type": "Method-Based",
            "rewritten_query": "Adapting RoBERTa for sentiment classification in user-generated content.",
            "relation_to_original": "Explores a similar task but applies a different transformer model (RoBERTa instead of BERT).",
            "relevance_score": 0.80,
        }
    ]
}
```

Ensure the output strictly follows the JSON format for easy parsing. Do not include extra explanations beyond the JSON output.
"""

template_query_list_expand = """You are an expert in academic search query refinement. Your task is to rewrite a given research query (or a list of queries) by maintaining its core intent while generating alternative queries that explore the same or similar research field or method. These rewritten queries should optimize search retrieval and enhance academic relevance. Additionally, provide a relevance score (ranging from 0 to 1) that quantifies the semantic similarity between the original query and each rewritten query.

### **Input:**
A raw academic search query or a list of queries provided by the user.

### **Rewriting Guidelines:**
1. **Field-Based Expansion**: Identify the core research domain of the query and generate alternative queries related to the same or closely related fields.
2. **Method-Based Variation**: If the query focuses on a methodology, generate alternatives using similar or related techniques.
3. **Terminology Enhancement**: Use precise academic terminology and relevant technical expressions to improve search effectiveness.
4. **Diversity of Queries**: Provide multiple rewritten queries that explore different but related aspects of the research field or method.
5. **Relevance Score**: Assign a similarity score (0 to 1) to each rewritten query based on its semantic closeness to the original query. A score of **1.0** indicates a near-identical meaning, while a score closer to **0.0** suggests lower relevance.

### **Output Format (for easy result analysis):**
Provide the results in structured JSON format as follows:
[
    {
        "original_query": "<user_input_query>",
        "rewritten_queries": [
            {
                "variation_type": "<Field-Based or Method-Based>",
                "rewritten_query": "<new_query_text>",
                "relation_to_original": "<brief_explanation_of_how_it_relates>",
                "relevance_score": <numeric_value_between_0_and_1>
            },
            {
                "variation_type": "<Field-Based or Method-Based>",
                "rewritten_query": "<new_query_text>",
                "relation_to_original": "<brief_explanation_of_how_it_relates>",
                "relevance_score": <numeric_value_between_0_and_1>
            }
        ]
    }
]

### **Example:**

Input: ["How does LoRA improve fine-tuning efficiency in NLP?", "What are the latest advancements in contrastive learning for computer vision?"]
Output:
[
    {
        "original_query": "How does LoRA improve fine-tuning efficiency in NLP?",
        "rewritten_queries": [
            {
                "variation_type": "Field-Based",
                "rewritten_query": "How does adapter-based fine-tuning enhance efficiency in NLP?",
                "relation_to_original": "Explores a broader category of adapter-based fine-tuning methods.",
                "relevance_score": 0.92
            },
            {
                "variation_type": "Method-Based",
                "rewritten_query": "Comparing LoRA and prompt tuning for efficient model adaptation in NLP.",
                "relation_to_original": "Includes an alternative fine-tuning technique (prompt tuning) for comparison.",
                "relevance_score": 0.85
            }
        ]
    },
    {
        "original_query": "What are the latest advancements in contrastive learning for computer vision?",
        "rewritten_queries": [
            {
                "variation_type": "Field-Based",
                "rewritten_query": "Recent progress in self-supervised contrastive learning for image recognition.",
                "relation_to_original": "Focuses on self-supervised contrastive learning within computer vision.",
                "relevance_score": 0.91
            }
        ]
    }
]

Ensure the output strictly follows the JSON format for easy parsing. Do not include extra explanations beyond the JSON output.
"""

template_extract_prompt = """You are an expert in **academic information retrieval**. Your task is to **split a user-provided query into multiple concise search queries** optimized for academic search engines such as **Openalex,Semantic Scholar, Google Scholar, and ArXiv**.
### **Input:**
A raw academic search query  provided by the user.

### **Guidelines for Query Splitting:**
1. **Extract Core Concepts**: Identify key topics, methodologies, and terminologies from the query.
2. **Generate Short Queries**: Break down the input into **short, meaningful, comma-separated queries** that capture the essence of the original query.
3. **Maintain Search Optimization**: Ensure the generated queries are effective for retrieving relevant academic papers by including alternative phrasings or synonyms.
4. **Format for Easy Parsing**: Return the results in a structured JSON format.

### **Output Format:**

{
    "original_query": "<user_input_query>",
    "search_query": "<short_query_1>, <short_query_2>, <short_query_3>, ..."
}


### **Example:**

Input: The impact of LoRA fine-tuning on transformer-based NLP models for low-resource languages.

Output:
{
    "original_query": "The impact of LoRA fine-tuning on transformer-based NLP models for low-resource languages.",
    "search_query": "LoRA fine-tuning, transformer NLP models, low-resource language NLP, LoRA for NLP, fine-tuning transformers"
}

Ensure that the output strictly follows this JSON structure without any extra explanations.
"""

template_extract_prompt_list = """You are an expert in **academic information retrieval and search query optimization**. Your task is to **split a given list of input queries** into concise, **short, and comma-separated search queries** that are well-structured for **academic paper retrieval**.

### **Guidelines:**
1. **Short & Concise**: The resulting search queries should use **essential academic keywords** while maintaining meaning.
2. **Comma-Separated**: Format the final search queries as **comma-separated phrases** for use in academic search engines.
3. **Query Optimization**:
   - Remove unnecessary words while preserving **technical accuracy**.
   - Ensure queries align with common academic **indexing and retrieval** best practices.
   - Maintain relevance for **academic search engines like Semantic Scholar, Google Scholar, and ArXiv**.
4. **Multiple Queries Handling**: If given multiple queries, process each separately and return structured results.

### **Output Format (for easy parsing):**

Return the result in **JSON format** as follows:

{
    "original_queries": ["<original_query_1>", "<original_query_2>", ...],
    "processed_queries": [
        {
            "original_query": "<original_query_1>",
            "search_queries": "<optimized_query_1>, <optimized_query_2>, <optimized_query_3>"
        },
        {
            "original_query": "<original_query_2>",
            "search_queries": "<optimized_query_1>, <optimized_query_2>, <optimized_query_3>"
        }
    ]
}


### **Example:**

Input:
[
    "How do retrieval-augmented LMs perform well in knowledge-intensive tasks?",
    "What are the latest advancements in multimodal large language models?"
]

Output:
{
    "original_queries": [
        "How do retrieval-augmented LMs perform well in knowledge-intensive tasks?",
        "What are the latest advancements in multimodal large language models?"
    ],
    "processed_queries": [
        {
            "original_query": "How do retrieval-augmented LMs perform well in knowledge-intensive tasks?",
            "search_queries": "retrieval-augmented LMs, knowledge-intensive tasks, large language models for knowledge-intensive tasks, retrieval-augmented generation"
        },
        {
            "original_query": "What are the latest advancements in multimodal large language models?",
            "search_queries": "multimodal large language models, vision-language models, multimodal transformers, cross-modal learning"
        }
    ]
}

Ensure that the output strictly follows this JSON structure without any extra explanations.

"""

template_expand_query_based_query_and_doc = """You are an expert in **academic search, information retrieval, and relevance assessment**. Your task is to analyze an **original search query** and the **content of a reference document**, determine their relevance, and **generate a refined academic search statement** if they are related. Additionally, provide a **clear explanation of relevance**, **extract search-friendly keywords**, and assign a **relevance score (0-1) between the original query and the new search query**.

### **Input:**
- **Original Query**: A user-provided search statement.
- **Reference Document**: Contains academic metadata including **title, authors, abstract, and field information**.


### **Processing Steps:**

1. **Relevance Determination**:
   - Compare the **semantic meaning** and **research focus** of the **original query** and the **reference document**.
   - If the document is **relevant**, proceed to generate a **new search statement**.
   - If the document is **not relevant**, do not generate a new query and return **relevance score (0-1)** is 0.
   - Assign a **relevance score (0-1)** that quantifies the similarity between the **original query** and the **new search query** (if applicable).

2. **New Search Statement Generation** (If Related):
   - Generate a refined **academic search query** that integrates insights from the reference document.
   - The new query should **expand or refine the original topic** while maintaining academic relevance.

3. **Keyword Extraction for Search Optimization** (If a New Query is Generated):
   - Extract **short, keyword-based phrases** from the new query.
   - Ensure keywords are **comma-separated** and suitable for use in academic search engines (e.g., **Semantic Scholar, Google Scholar, ArXiv**).

4. **Explanation of Relevance**:
   - Provide a **brief but clear explanation** of why the original query and the reference document are (or are not) related.


### **Output Format (Structured for Easy Parsing):**
Return the output in **JSON format** as follows:

{
    "relevance_assessment": {
        "is_related": "<true/false>",
        "explanation": "<reasoning_behind_relevance_decision>",
        "relevance_score": "<value_between_0_and_1>"
    },
    "new_search_query": "<generated_academic_search_query_if_applicable>",
    "search_keywords": "<comma_separated_keywords_if_applicable>"
}


### **Example Execution:**

#### **Input:**
{
    "original_query": "How do retrieval-augmented language models improve factual accuracy?",
    "reference_document": {
        "title": "Enhancing Knowledge Recall in Language Models with Retrieval-Augmented Generation",
        "authors": ["John Doe", "Jane Smith"],
        "abstract": "This paper explores retrieval-augmented generation (RAG) models, demonstrating their effectiveness in improving factual accuracy through external knowledge retrieval.",
        "fieldsOfStudy": "Natural Language Processing",
        "publicationYear": 2024
    }
}

#### **Output:**
{
    "relevance_assessment": {
        "is_related": "true",
        "explanation": "The reference document discusses retrieval-augmented generation (RAG), which directly relates to improving factual accuracy in language models.",
        "relevance_score": "0.92"
    },
    "new_search_query": "Retrieval-augmented generation for improving factual accuracy in language models",
    "search_keywords": "retrieval-augmented generation, factual accuracy, language models, knowledge retrieval"
}

Ensure the output strictly follows the JSON format for easy parsing. Do not include extra explanations beyond the JSON output.
"""

template_sim_between_query_doc = """You are an expert in **academic search, information retrieval, and relevance assessment**. Your task is to analyze a **search query** and a **reference academic article**, determine their **relevance**, and provide a **multidimensional assessment** considering various professional and academic perspectives.

#### **Input:**
- **Search Query:** A user-provided search statement.
- **Reference Document:** Contains **title, authors, abstract, citationsCount (total number of papers that references this paper), year (the year the paper was published) and field information (a list of the paper’s high-level academic categories from external sources)**.

#### **Evaluation Criteria:**
Assess the relevance with scores from 0.0 (completely irrelevant) to 1.0 (perfect match) based on:

1. **Conceptual Similarity**:
   - Does the document discuss the same or closely related concepts?
2. **Methodological Alignment**:
   - Does the paper employ research methods relevant to the query’s intent?
3. **Disciplinary Perspective**:
   - Does the document belong to the same field or an adjacent interdisciplinary domain?
4. **Problem-Solution Relevance**:
   - Does the article address the same research problem as the query implies?
5. **Citations and Influence**:
   - Is the document highly cited in the field relevant to the query?
6. **Temporal Relevance**:
   - Is the publication date suitable for addressing the latest research trends?

#### **Model Output Format:**
Return a structured JSON object with a **relevance score (0.0-1.0)** and detailed explanations for each evaluation criterion.

{
  "conceptual_similarity": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "methodological_alignment": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "disciplinary_perspective": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "problem_solution_relevance": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "citations_influence": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "temporal_relevance": { "score": 0.0-1.0, "explanation": "<simple and crucial>" }
}

### **Example**

Input:
{
  "query": "Deep reinforcement learning for robotic manipulation in dynamic environments",
  "reference_document": {
    "title": "A Survey on Deep Reinforcement Learning for Robotic Control",
    "authors": ["John Doe", "Jane Smith"],
    "abstract": "This paper provides a comprehensive survey of deep reinforcement learning (DRL) applications in robotic control. It covers various DRL algorithms, their adaptations for robotics, and challenges in real-world deployment.",
    "fieldsOfStudy": ["Artificial Intelligence", "Robotics"],
    "citationsCount": 2,
    "publicationYear": 2021,
  }
}

Output:
{
  "conceptual_similarity": {"score": 0.9, "explanation": "The document discusses DRL in robotics, which strongly aligns with the query." },
  "methodological_alignment": {"score": 0.7, "explanation": "The paper is a survey, summarizing DRL techniques rather than presenting new empirical findings on robotic manipulation in dynamic environments." },
  "disciplinary_perspective": {"score": 0.95, "explanation": "The document belongs to AI and Robotics, making it highly relevant to the query’s domain." },
  "problem_solution_relevance": {"score": 0.6, "explanation": "The paper covers general DRL applications but does not specifically focus on dynamic environments or robotic manipulation." },
  "citations_influence": {"score": 0.4, "explanation": "The document has only 2 citations, indicating limited influence in the field." },
  "temporal_relevance": {"score": 0.7, "explanation": "The paper was published in 2021. In the fast-evolving fields of artificial intelligence and robotics, it can still provide relevant and up-to-date information about deep reinforcement learning for robotic control, but it may not cover the very latest research trends." }
}

Ensure the output strictly follows the JSON format for easy parsing. Do not include extra explanations beyond the JSON output.
"""

template_sim_between_query_doc_v2_instruction = """You are an expert in **academic search, information retrieval, and relevance assessment**. Your task is to analyze a **search query** and a **reference academic article**, determine their **relevance**, and provide a **multidimensional assessment** considering various professional and academic perspectives.

#### **Input:**
- **Search Query:** A user-provided search statement.
- **Reference Document:** Contains **title, authors, abstract, citationsCount (total number of papers that references this paper), year (the year the paper was published) and field information (a list of the paper’s high-level academic categories from external sources)**.
- **Current Year:** {CURRENT_YEAR} (for assessing temporal relevance).

#### **Evaluation Criteria:**
Assess the relevance with scores from 0.0 (completely irrelevant) to 1.0 (perfect match) based on the following dimensions:
1. **Conceptual Similarity**: Does the document discuss the same or closely related concepts?
2. **Methodological Alignment**: Does the paper employ research methods relevant to the query’s intent?
3. **Disciplinary Perspective**: Does the document belong to the same field or an adjacent interdisciplinary domain?
4. **Problem-Solution Relevance**: Does the article address the same research problem as the query implies?
5. **Citations and Influence**: Is the document highly cited in the field relevant to the query?
6. **Temporal Relevance**: Is the publication date suitable given the current year and the field’s pace of change?

If any field in the reference document is missing (e.g., abstract or citationsCount), assign a lower score (e.g., 0.3) for the affected criterion and note the absence in the explanation.
"""

template_sim_between_query_doc_v2_format = """
#### **Model Output Format:**
Return a structured JSON object with a **relevance score (0.0-1.0)** and detailed, concise explanations for each criterion.
```json
{
  "conceptual_similarity": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "methodological_alignment": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "disciplinary_perspective": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "problem_solution_relevance": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "citations_influence": { "score": 0.0-1.0, "explanation": "<simple and crucial>" },
  "temporal_relevance": { "score": 0.0-1.0, "explanation": "<simple and crucial>" }
}
```

Ensure the output strictly follows the JSON format for easy parsing. Do not include extra explanations beyond the JSON output.

"""

template_expand_query_based_query_and_batch_doc = """You are an expert in academic literature search and query expansion. Given an original user query, a list of previously used search queries, and a set of relevant academic documents, your task is to generate **N diverse new queries** that help refine and expand the search. The newly generated queries should be relevant to the **user query** while ensuring **diversity** from the queries already used. Additionally, provide a similarity score between each new query and the user query and a brief explanation of how each query was derived.

### **Input Format:**
- **N**: The number of new queries to generate (e.g., 3).
- **user_query**: The original query input by the user.
- **search_query_list**: A list of queries that have already been used for searching.
- **relevance_doc**: A list of related academic articles, each containing the following fields:
  - `title`: The title of the paper
  - `authors`: The authors of the paper
  - `abstract`: A brief summary of the paper
  - `fieldofstudy`: The research domain of the paper
  - `publicationYear`: The publication year
  - *(other relevant metadata if available)*

### **Task Requirements:**
1. **Generate N new search queries** based on the **user_query** and insights from **relevance_doc**.
2. **Ensure diversity**: The newly generated queries should be meaningfully different from those in **search_query_list** to improve search coverage.
3. **For each new query, provide the following information**:
   - **query_text**: The generated query.
   - **similarity_score**: A numerical value (0-1) representing its similarity to **user_query** (1 being identical, 0 being completely unrelated).
   - **explanation**: A brief explanation of how the query was formed (e.g., inspired by specific fields of study, key terms from abstracts, alternative phrasing, etc.).

### **Output Format (JSON structured for easy parsing):**
```json
{
  "new_queries": [
    {
      "query_text": "<Generated query 1>",
      "similarity_score": 0.85,
      "explanation": "Derived from key terms in related papers focusing on [specific aspect]."
    },
    {
      "query_text": "<Generated query 2>",
      "similarity_score": 0.78,
      "explanation": "Inspired by alternative terminology in the field of [field name]."
    },
    ...
  ]
}
```

Ensure that the generated queries are **high-quality, relevant, and diverse** while maintaining a structured format for downstream processing.
"""

template_expand_query_based_query_and_batch_doc_v2 = """You are an expert in academic literature search and query expansion. Given an original user query, a list of previously used search queries, and a set of relevant academic documents, your task is to generate **N diverse new queries** that refine and expand the search. The new queries must be highly relevant to the **user_query**, leverage insights from **relevance_doc**, and ensure **diversity** from **search_query_list** by exploring different vocabulary, research perspectives, or methodologies. Provide a similarity score and explanation for each new query.

### **Input Format:**
- **user_query**: The original query input by the user (e.g., "machine learning for medical imaging").
- **search_query_list**: A list of queries already used (e.g., ["deep learning in medical diagnostics", "ML for radiology"]).
- **relevance_doc**: A list of related academic articles, each containing:
  - `title`: The title of the paper.
  - `authors`: The authors of the paper.
  - `abstract`: A brief summary of the paper.
  - `fieldofstudy`: The research domain of the paper.
  - `publicationYear`: The publication year.
- **N**: The number of new queries to generate (e.g., 3).

### **Task Requirements:**
1. **Generate N new search queries** by leveraging **user_query**, insights from **relevance_doc** (e.g., recent trends from publicationYear, interdisciplinary angles from fieldofstudy), and gaps in **search_query_list**.
2. **Ensure diversity**: New queries should differ from **search_query_list** in terms of vocabulary, research perspective, or methodology, maximizing search coverage. Analyze **search_query_list** to identify overused terms or unexplored aspects.
3. **For each new query, provide**:
   - **query_text**: The generated query.
   - **similarity_score**: A value (0-1) based on keyword overlap and semantic relevance to **user_query** (1 = identical, 0 = unrelated).
   - **explanation**: A brief explanation of how the query was formed (e.g., inspired by abstracts, fields of study, or alternative phrasing).

### **Output Format (JSON):**
```json
{
  "new_queries": [
    {
      "query_text": "<Generated query 1>",
      "similarity_score": 0.85,
      "explanation": "Derived from key terms in related papers focusing on [specific aspect]."
    },
    {
      "query_text": "<Generated query 2>",
      "similarity_score": 0.78,
      "explanation": "Inspired by alternative terminology in the field of [field name]."
    },
    ...
  ]
}
```

Ensure the generated queries are high-quality, relevant, and diverse, strictly following the JSON format.
"""

template_sim_between_query_doc_v2_inst = """### **Task:**
Evaluate whether the provided academic document (**Doc**) sufficiently satisfies the **scholarly paper search demand** of the **UserQuery**. The evaluation should consider **explicit and implicit relevance factors**, including **topic alignment, contextual meaning, depth of information**.

---

### **Input Data:**
- **Search Time:** {searchTime}
- **UserQuery:** {userQuery}
- **Document Details:**
  - **Title:** {Title}
  - **Abstract:** {Abstract}
  - **Author (if available):** {Author}
  - **fields Of Study (if available):** {fieldsOfStudy}
  - **Publication Year (if available):** {publicationYear}

---

### **Evaluation Criteria & Scoring (0-5 Scale):**
1. **Topic Match:** Does the document **explicitly** address the subject of the query? Consider keyword overlap, research area similarity, and whether the **core theme** aligns with the query.
2. **Contextual Relevance:** Does the document explore the **specific intent** or implicit aspects of the query?
3. **Depth & Completeness:** Does the document provide **in-depth analysis**, experimental results, or theoretical discussions that comprehensively address the query topic?

---
"""

template_sim_between_query_doc_v2_example = """
### **Output Format (Structured JSON):**
```json
{
  "topic_match": X,
  "contextual_relevance": X,
  "depth_completeness": X,
  "final_decision": "Relevant" / "Partially Relevant" / "Not Relevant",
  "justification": "Brief explanation of why the document does or does not meet the search demand."
}
```

---

### **Decision Guidelines:**
- **Relevant:** All scores **≥ 4**
- **Partially Relevant:** At least **two scores = 3**
- **Not Relevant:** Most scores **≤ 2**

Ensure the response strictly follows given JSON format for easy parsing. Do not include any extra text or explanations outside of this format.

**Output:**"""

template_rerank_doc = """You are an expert in academic research evaluation. Your task is to assess the **authority** and **timeliness** of a user’s search query: <{user_query}> and the recalled academic papers.

### **Evaluation Criteria:**

#### **1. Authority Score (0-1) – How reputable is the paper?**
Assign a **normalized score** between 0 and 1 based on:
- **Citation Count:** Higher citations indicate greater influence.
  - (≥1000 citations → 1.0, 500-999 → 0.8, 100-499 → 0.6, 10-99 → 0.4, <10 → 0.2).
- **Journal/Conference Reputation:** Is it a top-tier venue?
  - (Top-tier journal/conference → +0.2, mid-tier → +0.1, unknown → 0).
- **Author Credentials:** Are the authors well-cited experts?
  - (Highly cited authors → +0.2, moderately cited → +0.1, unknown → 0).

#### **2. Timeliness Score (0-1) – How relevant is the paper’s publication date?**
Assign a **normalized score** between 0 and 1 based on:
- **Publication Year:**
  - (Published in the last 2 years → 1.0, 3-5 years → 0.8, 6-10 years → 0.6, >10 years → 0.4).
- **Recent Citations (Last 5 Years):**
  - (≥100 recent citations in the last 5 years → +0.2, 50-99 → +0.1, <50 → 0).
- **Field-Specific Relevance:**
  - (Fast-changing fields like AI, biotech → prioritize recent papers).

### **Input:**
Each academic paper includes:
- **Title:** {title}
- **Authors:** {authors}
- **Publication Year:** <{pub_year}>
- **Journal/Conference:**: <{journal}>
- **Citation Count:** <{cite_count}>
- **Recent Citations (last 5 years):** <{five_years_cite_num}>

### **Output Format (Strict JSON Format):**
```json
{
    "authority_score": <float>,
    "timeliness_score": <float>,
    "evaluation": "<Brief justification for the assigned scores>"
}
```
---

### **Guidelines:**
- **Strictly adhere to JSON format** to ensure seamless parsing.
- **Use the scoring rules explicitly**—no subjective judgments.
- **Keep the evaluation concise** (max 2 sentences).
- **Do not include extra explanations** beyond the required output.
---
**Output:**"""

template_from_pasa = """You are an expert in academic research. Given a query and a document in the context of a scholarly paper search, evaluate their relevance on a scale from 0 to 1, where 0 means completely irrelevant and 1 means highly relevant. Base your evaluation on the query’s intent, key concepts, and the document’s content. Provide a score and explain your reasoning consistently.

Query: {user_query}
Document:
Title: {title}
Abstract: {abstract}

Score: [Your score between 0 and 1]
Reasoning: [Your explanation]
"""

template_from_pasa_refine = """You are an expert in academic information retrieval and relevance assessment. Given a user query and a scholarly document (title and abstract), assess how relevant the document is to the query on a scale from 0 to 1:

* 0 means completely irrelevant.
* 1 means highly relevant.

Your assessment should be based on the intent of the query, the key concepts it expresses, and how well these are addressed or reflected in the document. Return both:

* A numerical score (between 0 and 1).
* A concise and well-reasoned explanation that justifies the score.

Use consistent reasoning criteria across evaluations.

Query: {user_query}

Document:
  Title: {title}
  Abstract: {abstract}

Score: [Your score between 0 and 1]
Reasoning: [Justification of the score, considering query intent, concept alignment, and document content]
"""

evaluation_prompt = """
You are a professional academic writing assistant. Please evaluate the similarity between the user’s content and the article content, and provide a relevance score between 0 and 1.

### **Input_format**
Question: A raw academic search query  provided by the user.
Article:
    - Title: Title of the academic article
    - Author: All authors of the academic article
    - Year: Publication date of the academic article
    - Abstract: Abstract of the academic article, explaining and summarizing the content

---
### **Evaluation Criteria & Scoring (0-1 Scale):**
- Topic Match: Does the document explicitly address the subject of the query? Consider keyword overlap, research area similarity, and alignment with the core theme of the query.
- Contextual Relevance: Does the document explore the specific intent or implicit aspects of the query?
- Depth & Completeness: Does the document provide in-depth analysis, experimental results, or theoretical discussions that comprehensively address the query topic?
**Score Ranges:**
0.0 - 0.09: Completely unrelated. The document does not address the core content of the question at all, or it completely deviates from the topic.
0.1 - 0.49: Very low relevance. The document addresses a very limited aspect of the topic or is mostly irrelevant, containing only minor parts that may be loosely connected.
0.4 - 0.59: Low relevance. The document contains some relevant content but fails to fully answer the question, or some information is incorrect or incomplete.
0.6 - 0.79: Moderate relevance. The document covers the main aspects of the question but lacks certain details or depth, or there may be some deviations in the content.
0.8 - 0.99: High relevance. The document broadly covers the key points of the question, is mostly accurate and complete, but may lack minor details or have slight deviations.
1: Perfect relevance. The document completely and accurately answers the question, covering all core aspects with complete information and no deviations.
---
### **Note:**
- Provide only a numerical score without analysis.
- Ensure the score precision is up to two decimal places.
- Do not provide vague or overly broad scores. Ensure the score directly reflects the content’s relevance.
---
### **Example:**
Question: What are the latest methods for enhancing the clarity and realism of image generation models?
Article:
    - Title: "Progressive Knowledge Distillation of Stable Diffusion XL Using Layer-Level Loss"
    - Author: "Gupta, Yatharth; Jaddipal, Vishnu V.; Prabhala, Harish; Paul, Sayak; Von Platen, Patrick"
    - Year: "2024"
    - Abstract:
    - "Stable Diffusion XL (SDXL) has become the best open-source text-to-image model (T2I) for its versatility and top-notch image quality. Efficiently addressing the computational demands of SDXL models is crucial for wider reach and applicability. In this work, we introduce two scaled-down variants, Segmind Stable Diffusion (SSD-1B) and Segmind-Vega, with 1.3B and 0.74B parameter UNets, respectively, achieved through progressive removal using layer-level losses focusing on reducing the model size while preserving generative quality. We release these model weights at https://hf.co/Segmind. Our methodology involves the elimination of residual networks and transformer blocks from the U-Net structure of SDXL, resulting in significant reductions in parameters, and latency. Our compact models effectively emulate the original SDXL by capitalizing on transferred knowledge, achieving competitive results against larger multi-billion parameter SDXL. Our work underscores the efficacy of knowledge distillation coupled with layer-level losses in reducing model size while preserving the high-quality generative capabilities of SDXL, thus facilitating more accessible deployment in resource-constrained environments."

Output: 0.82
---

### **Input Data:**
Question: "{query}"
Article:
    - Title: "{title}"
    - Author: "{author}"
    - Year: "{year}"
    - Abstract: "{abstract}"
Please evaluate the similarity based on the criteria above and output a score between 0 and 1, indicating the relevance of the answer to the question.

**Output:**"""


template_eval_evoled = """You are a rigorous and highly discerning academic search relevance evaluator. Your task is to critically assess the relationship between the user's query and the provided scholarly article. Apply a strict, high-standard academic lens to evaluate conceptual alignment, topical focus, and methodological relevance. Be skeptical of superficial keyword matches or loosely related themes. Only assign a high relevance score (on a 0–1 scale) when there is clear and substantial alignment in research purpose, methods, and contribution. Err on the side of conservatism in scoring—precision and selectivity are paramount.

### **Input Format**
Query: Raw academic search query
Article:
    - Title: Academic article title
    - Abstract: Abstract text summarizing the paper's content

---
### **Hierarchical Evaluation Protocol**

**1. Critical Relevance Check (Binary Gate)**
- If the document contains ZERO of the following, automatically score 0.0:
  - Core subject keywords from query
  - Matching research domain
  - Thematic alignment with query intent

**2. Detailed Scoring Criteria** *(Only if passes Critical Check)*

A. **Core Topic Alignment (0-0.6)**
- 0.5-0.6: Directly addresses primary subject with matching terminology
- 0.3-0.4: Related subfield but different focus area
- 0.1-0.2: Only tangential connection through peripheral terms
- 0.0: Fails Critical Relevance Check

B. **Contextual Precision (0-0.3)**
- 0.2-0.3: Explicitly addresses query's specific technical aspects
- 0.1: General thematic similarity without concrete details
- 0.0: No meaningful connection to query intent

C. **Depth Validation (0-0.1)**
- 0.1: Provides experimental validation/novel theoretical framework
- 0.05: Mentions concept without substantive analysis
- 0.0: Superficial treatment of subject

---
### **Scoring Matrix** *(Sum Components A+B+C)*
0.00-0.19: Completely irrelevant/off-topic
0.20-0.39: Minimal relevance - shares domain but different focus
0.40-0.59: Partial relevance - addresses some aspects
0.60-0.79: Substantial relevance - covers key elements
0.80-1.00: Optimal match - comprehensive coverage

---
### **Anti-Gaming Rules**
- Penalize -0.3 for keyword stuffing without contextual relevance
- Penalize -0.2 for misleading titles/abstracts
- If score <0.4, round down to nearest 0.1
- If score ≥0.7, require positive marks in all 3 criteria

---
### **Examples**

**Example 1 (Low Score)**
Query: "Machine learning for early Alzheimer's diagnosis using MRI"
Article: "Statistical analysis of MRI machine calibration errors"
Reasoning: Fails Critical Relevance - no ML or Alzheimer's content
Score: 0.15

**Example 2 (High Score)**
Query: "Federated learning optimization in IoT networks"
Article: "Adaptive Gradient Compression for Energy-Efficient Federated Learning in Edge Computing Environments"
Reasoning: Directly addresses FL optimization (0.6) + technical specifics (0.25) + experimental validation (0.1)
Score: 0.86


---
### **Input Data**
Query: {query}
Article: {doc}

---
### **Output format:**
**Reasoning:** [Concise technical justification]
**Score:** [0.00-1.00]
---
"""
