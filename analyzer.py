"""
AI-Powered Documentation Analyzer - Enhanced Version
With marketer-focused analysis and improved technical features
"""
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import sys
import re
import os
import os
import time
import random
import textstat  # For advanced readability metrics
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Configuration constants
MAX_CONTENT_LENGTH = 15000  # Limit content size for analysis
MARKETER_JARGON = [
    'cta', 'conversion', 'funnel', 'kpi', 'roi', 'segmentation', 
    'persona', 'engagement', 'retention', 'acquisition', 'crm',
    'lead gen', 'conversion rate', 'a/b test', 'customer journey'
]
TECHNICAL_JARGON = [
    'api', 'sdk', 'endpoint', 'json', 'xml', 'http', 'https',
    'authentication', 'oauth', 'webhook', 'snippet', 'integration'
]

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
# Try importing AI libraries with fallbacks
HF_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
    print("✅ Hugging Face transformers available")
except ImportError as e:
    print(f"❌ Hugging Face transformers not available: {e}")

try:
    import openai
    OPENAI_AVAILABLE = True
    print("✅ OpenAI library available")
except ImportError:
    print("❌ OpenAI library not available")

@dataclass
class AnalysisResult:
    """Enhanced data structure to hold analysis results with confidence scores"""
    url: str
    readability: Dict[str, Any]
    structure: Dict[str, Any]
    completeness: Dict[str, Any]
    style_guidelines: Dict[str, Any]
    marketer_focus: Dict[str, Any]
    analyzer_type: str = "rule-based"
    confidence_scores: Dict[str, float] = None

class DocumentationAnalyzer:
    """Enhanced AI-powered documentation analyzer with marketer focus"""
    
    def __init__(self, analyzer_type: str = "auto", model_name: Optional[str] = None, 
             openai_api_key: Optional[str] = None):
        """
        Initialize the analyzer with enhanced configuration
        """
        self.analyzer_type = analyzer_type
        self.model_name = model_name
        self.generator = None
        self.openai_client = None
        self.content_cache = {}
        
        # Setup session for web requests
        self.session = self._setup_session()
        
        # Initialize AI backend
        self._initialize_ai_backend(analyzer_type, model_name, openai_api_key)
    
    def _initialize_ai_backend(self, analyzer_type: str, model_name: Optional[str], 
                             openai_api_key: Optional[str]):
        """Initialize the AI backend with better error handling"""
        
        if analyzer_type == "auto":
            if HF_AVAILABLE:
                analyzer_type = "huggingface"
                model_name = model_name or "facebook/bart-large-cnn"
            elif OPENAI_AVAILABLE and openai_api_key:
                analyzer_type = "openai"
            else:
                analyzer_type = "rule-based"
        
        self.analyzer_type = analyzer_type
        
        if analyzer_type == "huggingface" and HF_AVAILABLE:
            self._setup_huggingface(model_name)
        elif analyzer_type == "openai" and OPENAI_AVAILABLE:
            self._setup_openai(openai_api_key)
        else:
            self.analyzer_type = "rule-based"
            print("Using rule-based analysis (no AI libraries available)")
    
    def _setup_huggingface(self, model_name: Optional[str]):
        """Setup Hugging Face model with better defaults"""
        try:
            print(f"Loading Hugging Face model: {model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                framework="pt"
            )
            self.model_name = model_name
            print(f"✅ Hugging Face model loaded: {model_name}")
            
        except Exception as e:
            print(f"❌ Error loading Hugging Face model: {e}")
            self.analyzer_type = "rule-based"
    
    def _setup_openai(self, api_key: Optional[str]):
        """Setup OpenAI client with better configuration"""
        try:
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment")
            
            self.openai_client = openai.OpenAI(
                api_key=api_key,
                timeout=30.0,
                max_retries=3
            )
            print("✅ OpenAI client initialized")
            
        except Exception as e:
            print(f"❌ Error setting up OpenAI: {e}")
            self.analyzer_type = "rule-based"
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with enhanced retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1.5,
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=100
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        })
        
        return session
    
    def fetch_content(self, url: str) -> Dict[str, str]:
        """Enhanced content fetcher with MoEngage-specific handling"""
        try:
            print(f"Fetching content from: {url}")
            time.sleep(random.uniform(1.5, 3.5))
            
            response = self.session.get(url, timeout=45)
            response.raise_for_status()
            
            if 'x-moengage-platform' in response.headers:
                print("Detected MoEngage platform content")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title found"
            
            moengage_selectors = [
                'div.article-content',
                'div.help-article-content',
                'div.md-content__inner'
            ]
            
            content_selectors = moengage_selectors + [
                'article', 
                'main article',
                '.documentation-content',
                '.content-inner',
                '#content-main',
                '#main-content',
                'div.content',
                'div.main',
                'body'
            ]
            
            content = None
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0]
                    print(f"Using selector: {selector}")
                    break
            
            metadata = {
                'categories': [],
                'tags': [],
                'last_updated': None
            }
            
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                if tag.get('property', '').startswith('article:'):
                    key = tag['property'].replace('article:', '')
                    metadata[key] = tag.get('content', '')
            
            if content:
                for elem in content(["script", "style", "nav", "footer", 
                                   "header", "aside", "iframe", "form",
                                   "button", "div.advertisement"]):
                    elem.decompose()
                
                content_text = content.get_text(separator='\n', strip=True)
                content_text = re.sub(r'\n\s*\n', '\n\n', content_text)
                content_text = re.sub(r'\n{3,}', '\n\n', content_text)
                
                headings = []
                for level in range(1, 7):
                    for heading in content.find_all(f'h{level}'):
                        headings.append({
                            'level': level,
                            'text': heading.get_text().strip(),
                            'id': heading.get('id', '')
                        })
                
                code_blocks = []
                for pre in content.find_all('pre'):
                    code_blocks.append(pre.get_text().strip())
                
                images = []
                for img in content.find_all('img', src=True):
                    images.append({
                        'src': img['src'],
                        'alt': img.get('alt', ''),
                        'title': img.get('title', '')
                    })
            else:
                content_text = "Could not extract main content"
                headings = []
                code_blocks = []
                images = []
            
            return {
                "title": title_text,
                "content": content_text[:MAX_CONTENT_LENGTH],
                "raw_html": str(content)[:MAX_CONTENT_LENGTH] if content else "",
                "metadata": metadata,
                "headings": headings,
                "code_blocks": code_blocks,
                "images": images,
                "url": url
            }
            
        except Exception as e:
            print(f"Error fetching content from {url}: {str(e)}")
            return {
                "title": "Error",
                "content": f"Error fetching content: {str(e)}",
                "raw_html": "",
                "metadata": {},
                "headings": [],
                "code_blocks": [],
                "images": [],
                "url": url
            }
    
    def analyze_readability(self, content: str) -> Dict[str, Any]:
        """Enhanced readability analysis with marketer focus"""
        flesch = textstat.flesch_reading_ease(content)
        grade_level = textstat.flesch_kincaid_grade(content)
        gunning_fog = textstat.gunning_fog(content)
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        
        content_lower = content.lower()
        tech_jargon_count = sum(content_lower.count(term) for term in TECHNICAL_JARGON)
        marketing_jargon_count = sum(content_lower.count(term) for term in MARKETER_JARGON)
        
        if grade_level <= 8 and tech_jargon_count <= 5:
            readability_score = "Excellent"
            confidence = 0.9
        elif grade_level <= 10 and tech_jargon_count <= 10:
            readability_score = "Good"
            confidence = 0.7
        elif grade_level <= 12 or marketing_jargon_count >= 5:
            readability_score = "Fair"
            confidence = 0.6
        else:
            readability_score = "Poor"
            confidence = 0.8
        
        marketer_issues = []
        if tech_jargon_count > 5:
            marketer_issues.append(f"High technical jargon ({tech_jargon_count} instances)")
        if marketing_jargon_count < 3:
            marketer_issues.append("Low use of marketing terminology")
        if avg_sentence_length > 20:
            marketer_issues.append(f"Average sentence length ({avg_sentence_length:.1f} words) too long")
        
        problematic_sentences = []
        for i, sentence in enumerate(long_sentences[:3]):
            problematic_sentences.append({
                "sentence": sentence[:150] + ("..." if len(sentence) > 150 else ""),
                "word_count": len(sentence.split()),
                "position": f"Paragraph {i+1}"
            })
        
        return {
            "readability_score": readability_score,
            "confidence": confidence,
            "metrics": {
                "flesch_reading_ease": flesch,
                "grade_level": grade_level,
                "gunning_fog_index": gunning_fog,
                "avg_sentence_length": avg_sentence_length,
                "technical_jargon_count": tech_jargon_count,
                "marketing_jargon_count": marketing_jargon_count
            },
            "marketer_specific": {
                "assessment": "Good for marketers" if marketing_jargon_count >= 3 and tech_jargon_count <= 8 else "Needs adaptation for marketers",
                "issues": marketer_issues,
                "jargon_balance": {
                    "technical": tech_jargon_count,
                    "marketing": marketing_jargon_count,
                    "recommendation": "Increase marketing terms" if marketing_jargon_count < 3 else "OK"
                }
            },
            "problematic_sentences": problematic_sentences,
            "suggestions": [
                f"Simplify sentences over 25 words (found {len(long_sentences)})",
                "Add definitions for technical terms" if tech_jargon_count > 5 else "",
                "Include more marketing-focused examples" if marketing_jargon_count < 3 else "",
                f"Break up long sentences (e.g., '{long_sentences[0][:50]}...')" if long_sentences else ""
            ]
        }
    
    def analyze_structure(self, content: str, raw_html: str, headings: List[Dict]) -> Dict[str, Any]:
        """Enhanced structure analysis with heading hierarchy checks"""
        heading_levels = [h['level'] for h in headings]
        hierarchy_issues = []
        
        for i in range(len(heading_levels)-1):
            if heading_levels[i+1] > heading_levels[i] + 1:
                hierarchy_issues.append(
                    f"Heading jump from h{heading_levels[i]} to h{heading_levels[i+1]} "
                    f"between '{headings[i]['text'][:30]}...' and '{headings[i+1]['text'][:30]}...'"
                )
        
        if heading_levels and heading_levels[0] != 1:
            hierarchy_issues.append(f"Document starts with h{heading_levels[0]} instead of h1")
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]
        
        list_items = re.findall(r'^\s*[\-*•]\s+.+$', content, re.MULTILINE)
        numbered_lists = re.findall(r'^\s*\d+\.\s+.+$', content, re.MULTILINE)
        
        if not hierarchy_issues and len(headings) >= 3 and len(list_items) >= 3:
            structure_score = "Excellent"
            confidence = 0.85
        elif len(hierarchy_issues) <= 2 and len(headings) >= 2:
            structure_score = "Good"
            confidence = 0.7
        else:
            structure_score = "Poor"
            confidence = 0.8
        
        return {
            "structure_score": structure_score,
            "confidence": confidence,
            "heading_analysis": {
                "total_headings": len(headings),
                "heading_levels": heading_levels,
                "hierarchy_issues": hierarchy_issues,
                "missing_h1": heading_levels[0] != 1 if heading_levels else True
            },
            "paragraph_analysis": {
                "total_paragraphs": len(paragraphs),
                "avg_paragraph_length": avg_paragraph_length,
                "long_paragraphs": len(long_paragraphs),
                "example_long_paragraph": long_paragraphs[0][:100] + "..." if long_paragraphs else None
            },
            "list_analysis": {
                "bullet_lists": len(list_items),
                "numbered_lists": len(numbered_lists),
                "recommended_lists": "Add more lists" if len(list_items) + len(numbered_lists) < 3 else "Sufficient"
            },
            "suggestions": [
                f"Fix heading hierarchy: {issue}" for issue in hierarchy_issues[:3]
            ] + [
                f"Break up long paragraphs (found {len(long_paragraphs)} over 100 words)",
                "Add more bulleted lists" if len(list_items) < 3 else "",
                "Ensure document starts with H1" if heading_levels and heading_levels[0] != 1 else ""
            ]
        }
    
    def analyze_completeness(self, content_data: Dict) -> Dict[str, Any]:
        """Enhanced completeness analysis with marketing focus"""
        content = content_data['content']
        metadata = content_data['metadata']
        code_blocks = content_data['code_blocks']
        images = content_data['images']
        
        has_prerequisites = bool(re.search(r'prereq|requirement|before you begin', content, re.I))
        has_steps = bool(re.search(r'step\s+\d+|first|next|then|finally', content, re.I))
        has_examples = len(code_blocks) > 0 or bool(re.search(r'example[s]?|for instance', content, re.I))
        has_screenshots = len([img for img in images if 'screen' in img['alt'].lower() or 'screen' in img['title'].lower()]) > 0
        has_troubleshooting = bool(re.search(r'troubleshoot|common issues|error', content, re.I))
        has_use_cases = bool(re.search(r'use case|scenario|marketing application', content, re.I))
        has_roi = bool(re.search(r'roi|return on investment|business impact', content, re.I))
        has_metrics = bool(re.search(r'metric|kpi|measure|analytics', content, re.I))
        
        completeness_factors = sum([
            has_prerequisites,
            has_steps,
            has_examples,
            has_screenshots,
            has_troubleshooting,
            has_use_cases,
            has_roi,
            has_metrics
        ])
        
        if completeness_factors >= 7:
            completeness_score = "Excellent"
            confidence = 0.9
        elif completeness_factors >= 5:
            completeness_score = "Good"
            confidence = 0.75
        elif completeness_factors >= 3:
            completeness_score = "Fair"
            confidence = 0.6
        else:
            completeness_score = "Poor"
            confidence = 0.8
        
        missing_elements = []
        if not has_prerequisites:
            missing_elements.append("Prerequisites section")
        if not has_steps:
            missing_elements.append("Step-by-step instructions")
        if not has_examples:
            missing_elements.append("Practical examples")
        if not has_screenshots:
            missing_elements.append("Screenshots or diagrams")
        if not has_troubleshooting:
            missing_elements.append("Troubleshooting section")
        if not has_use_cases:
            missing_elements.append("Marketing use cases")
        if not has_roi:
            missing_elements.append("ROI/business impact explanation")
        if not has_metrics:
            missing_elements.append("Key metrics/KPIs")
        
        return {
            "completeness_score": completeness_score,
            "confidence": confidence,
            "content_coverage": {
                "prerequisites": has_prerequisites,
                "step_by_step": has_steps,
                "examples": has_examples,
                "visual_aids": has_screenshots,
                "troubleshooting": has_troubleshooting,
                "marketing_use_cases": has_use_cases,
                "business_impact": has_roi,
                "metrics_tracking": has_metrics
            },
            "missing_elements": missing_elements,
            "marketing_specific": {
                "use_cases_present": has_use_cases,
                "business_context_provided": has_roi,
                "metrics_mentioned": has_metrics,
                "assessment": "Good marketing focus" if has_use_cases and has_roi else "Needs more marketing perspective"
            },
            "suggestions": [
                f"Add {element}" for element in missing_elements[:4]
            ] + [
                "Include real-world marketing scenarios" if not has_use_cases else "",
                "Show business impact metrics" if not has_roi else "",
                "Add dashboard/screenshot examples" if not has_screenshots else ""
            ]
        }
    
    def analyze_style_guidelines(self, content: str) -> Dict[str, Any]:
        """Enhanced style analysis with Microsoft Style Guide checks"""
        passive_indicators = [' is ', ' are ', ' was ', ' were ', ' been ', ' being ']
        passive_count = sum(content.lower().count(indicator) for indicator in passive_indicators)
        
        action_words = ['click', 'select', 'choose', 'enter', 'type', 'navigate', 
                       'follow', 'complete', 'configure', 'set up', 'create']
        action_count = sum(content.lower().count(word) for word in action_words)
        
        second_person = content.lower().count(' you ') + content.lower().count(' your ')
        
        terms = re.findall(r'\b(API|SDK|integration|endpoint)\b', content, re.I)
        term_variations = len(set(term.lower() for term in terms)) if terms else 0
        
        style_factors = sum([
            passive_count < 10,
            action_count >= 5,
            second_person >= 5,
            term_variations <= 2
        ])
        
        if style_factors == 4:
            style_score = "Excellent"
            confidence = 0.85
        elif style_factors >= 2:
            style_score = "Good"
            confidence = 0.7
        else:
            style_score = "Poor"
            confidence = 0.8
        
        return {
            "style_score": style_score,
            "confidence": confidence,
            "voice_and_tone": {
                "passive_voice_instances": passive_count,
                "action_oriented_commands": action_count,
                "second_person_usage": second_person,
                "assessment": "Too passive" if passive_count > 10 else "Appropriately active"
            },
            "clarity_and_conciseness": {
                "term_consistency": f"{len(terms)} instances with {term_variations} variations",
                "assessment": "Consistent terminology" if term_variations <= 2 else "Inconsistent terms"
            },
            "suggestions": [
                f"Reduce passive voice (found {passive_count} instances)",
                f"Increase action verbs (found {action_count})" if action_count < 5 else "",
                "Use more 'you' and 'your' for direct address" if second_person < 5 else "",
                f"Standardize terminology (found {term_variations} variations)" if term_variations > 2 else ""
            ]
        }
    
    def analyze_marketer_focus(self, content_data: Dict) -> Dict[str, Any]:
        """Specialized analysis for marketing audience"""
        content = content_data['content']
        metadata = content_data['metadata']
        
        jargon_count = sum(content.lower().count(term) for term in MARKETER_JARGON)
        tech_jargon_count = sum(content.lower().count(term) for term in TECHNICAL_JARGON)
        
        has_campaign = bool(re.search(r'campaign|promotion', content, re.I))
        has_audience = bool(re.search(r'audience|segment|persona', content, re.I))
        has_metrics = bool(re.search(r'metric|kpi|measure|analytics', content, re.I))
        has_roi = bool(re.search(r'roi|return on investment|business impact', content, re.I))
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        tech_sentences = sum(1 for s in sentences if any(term in s.lower() for term in TECHNICAL_JARGON))
        marketing_sentences = sum(1 for s in sentences if any(term in s.lower() for term in MARKETER_JARGON))
        focus_ratio = marketing_sentences / max(tech_sentences, 1)
        
        marketing_factors = sum([
            jargon_count >= 5,
            has_campaign,
            has_audience,
            has_metrics,
            has_roi,
            focus_ratio >= 0.5
        ])
        
        if marketing_factors >= 5:
            score = "Excellent"
            confidence = 0.9
        elif marketing_factors >= 3:
            score = "Good"
            confidence = 0.75
        else:
            score = "Poor"
            confidence = 0.8
        
        return {
            "marketer_score": score,
            "confidence": confidence,
            "marketing_jargon": jargon_count,
            "technical_jargon": tech_jargon_count,
            "focus_ratio": f"{focus_ratio:.1f} (marketing:technical)",
            "content_focus": {
                "campaign_mentioned": has_campaign,
                "audience_segmentation": has_audience,
                "metrics_included": has_metrics,
                "roi_discussed": has_roi
            },
            "issues": [
                "Too technical" if focus_ratio < 0.3 else "",
                "Lacks marketing context" if not has_roi else "",
                "Needs more campaign examples" if not has_campaign else ""
            ],
            "suggestions": [
                "Add marketing campaign examples" if not has_campaign else "",
                "Include audience segmentation guidance" if not has_audience else "",
                "Show ROI calculations" if not has_roi else "",
                "Balance technical details with marketing value propositions"
            ]
        }
    
    def analyze_document(self, url: str) -> AnalysisResult:
        """Enhanced main analysis method with confidence scores"""
        print(f"Starting analysis of: {url}")
        
        content_data = self.fetch_content(url)
        
        readability = self.analyze_readability(content_data['content'])
        structure = self.analyze_structure(
            content_data['content'],
            content_data['raw_html'],
            content_data['headings']
        )
        completeness = self.analyze_completeness(content_data)
        style = self.analyze_style_guidelines(content_data['content'])
        marketer_focus = self.analyze_marketer_focus(content_data)
        
        confidences = {
            "readability": readability.get("confidence", 0.7),
            "structure": structure.get("confidence", 0.7),
            "completeness": completeness.get("confidence", 0.7),
            "style": style.get("confidence", 0.7),
            "marketer_focus": marketer_focus.get("confidence", 0.7)
        }
        overall_confidence = sum(confidences.values()) / len(confidences)
        
        return AnalysisResult(
            url=url,
            readability=readability,
            structure=structure,
            completeness=completeness,
            style_guidelines=style,
            marketer_focus=marketer_focus,
            analyzer_type=self.analyzer_type,
            confidence_scores=confidences
        )

    def generate_report(self, result: AnalysisResult, format: str = "text") -> str:
        """Generate a formatted report from analysis results"""
        report = []
        
        if format == "text":
            report.append(f"Documentation Analysis Report for: {result.url}")
            report.append(f"Analysis Type: {result.analyzer_type}")
            report.append("\n=== Readability ===")
            report.append(f"Score: {result.readability['readability_score']} (Confidence: {result.readability['confidence']:.1f})")
            report.append(f"- Flesch Reading Ease: {result.readability['metrics']['flesch_reading_ease']:.1f}")
            report.append(f"- Grade Level: {result.readability['metrics']['grade_level']:.1f}")
            report.append(f"- Technical Jargon: {result.readability['metrics']['technical_jargon_count']}")
            report.append(f"- Marketing Terms: {result.readability['metrics']['marketing_jargon_count']}")
            report.append("Suggestions:")
            for suggestion in result.readability['suggestions']:
                if suggestion: report.append(f"  - {suggestion}")
            
            report.append("\n=== Structure ===")
            report.append(f"Score: {result.structure['structure_score']} (Confidence: {result.structure['confidence']:.1f})")
            report.append(f"- Headings: {result.structure['heading_analysis']['total_headings']}")
            report.append(f"- Hierarchy Issues: {len(result.structure['heading_analysis']['hierarchy_issues'])}")
            report.append(f"- Long paragraphs: {result.structure['paragraph_analysis']['long_paragraphs']}")
            report.append("Suggestions:")
            for suggestion in result.structure['suggestions']:
                if suggestion: report.append(f"  - {suggestion}")
            
            report.append("\n=== Completeness ===")
            report.append(f"Score: {result.completeness['completeness_score']} (Confidence: {result.completeness['confidence']:.1f})")
            report.append("Coverage:")
            for k, v in result.completeness['content_coverage'].items():
                report.append(f"- {k.replace('_', ' ').title()}: {'✅' if v else '❌'}")
            report.append("Suggestions:")
            for suggestion in result.completeness['suggestions']:
                if suggestion: report.append(f"  - {suggestion}")
            
            report.append("\n=== Style Guidelines ===")
            report.append(f"Score: {result.style_guidelines['style_score']} (Confidence: {result.style_guidelines['confidence']:.1f})")
            report.append(f"- Passive Voice: {result.style_guidelines['voice_and_tone']['passive_voice_instances']}")
            report.append(f"- Action Verbs: {result.style_guidelines['voice_and_tone']['action_oriented_commands']}")
            report.append("Suggestions:")
            for suggestion in result.style_guidelines['suggestions']:
                if suggestion: report.append(f"  - {suggestion}")
            
            report.append("\n=== Marketer Focus ===")
            report.append(f"Score: {result.marketer_focus['marketer_score']} (Confidence: {result.marketer_focus['confidence']:.1f})")
            report.append(f"- Marketing to Technical Ratio: {result.marketer_focus['focus_ratio']}")
            report.append("Content Focus:")
            for k, v in result.marketer_focus['content_focus'].items():
                report.append(f"- {k.replace('_', ' ').title()}: {'✅' if v else '❌'}")
            report.append("Suggestions:")
            for suggestion in result.marketer_focus['suggestions']:
                if suggestion: report.append(f"  - {suggestion}")
            
            return "\n".join(report)
        
        elif format == "json":
            return json.dumps({
                "url": result.url,
                "analyzer_type": result.analyzer_type,
                "confidence_scores": result.confidence_scores,
                "readability": result.readability,
                "structure": result.structure,
                "completeness": result.completeness,
                "style_guidelines": result.style_guidelines,
                "marketer_focus": result.marketer_focus
            }, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

def main():
    """Command line interface for the analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Powered Documentation Analyzer")
    parser.add_argument("url", help="URL of the documentation to analyze")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                      help="Output format (text or json)")
    parser.add_argument("--analyzer", choices=["auto", "huggingface", "openai", "rule-based"],
                      default="auto", help="Analysis method to use")
    parser.add_argument("--model", help="Hugging Face model name")
    parser.add_argument("--openai-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    analyzer = DocumentationAnalyzer(
        analyzer_type=args.analyzer,
        model_name=args.model,
        openai_api_key=args.openai_key
    )
    
    result = analyzer.analyze_document(args.url)
    report = analyzer.generate_report(result, args.format)
    
    print(report)

if __name__ == "__main__":
    main()