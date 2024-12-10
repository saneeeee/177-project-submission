# washdetector/agent/wash_detector.py
from typing import List, Dict
import openai
from pydantic import BaseModel
from .prompts import create_analysis_prompt
from ..metrics.compute import compute_all_metrics

class WashTradingAnalysis(BaseModel):
    """Structured output from the wash trading analysis"""
    reasoning_process: Dict[str, str]  # Step-by-step reasoning
    evidence: Dict[str, List[str]]     # Evidence per category
    alternative_explanations: List[str]
    is_wash_trading: bool
    confidence: float
    suspicious_addresses: List[str]
    recommendations: List[str]
    raw_response: str

class WashTradeDetector:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the reasoning text"""
        try:
            return text.split(section_name + ":")[1].split("EVIDENCE SUMMARY")[0].strip()
        except IndexError:
            try:
                return text.split(section_name + ":")[1].split("\n\n")[0].strip()
            except IndexError:
                return ""

    def _parse_evidence_sections(self, evidence_text: str) -> Dict[str, List[str]]:
        """Parse evidence into categorized sections"""
        evidence = {
            'network': [],
            'volume': [],
            'temporal': [],
            'price': [],
            'behavioral': []
        }
        
        current_section = None
        for line in evidence_text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "Network Evidence" in line:
                current_section = 'network'
            elif "Volume Evidence" in line:
                current_section = 'volume'
            elif "Temporal Evidence" in line:
                current_section = 'temporal'
            elif "Price Evidence" in line:
                current_section = 'price'
            elif "Behavioral Evidence" in line:
                current_section = 'behavioral'
            elif current_section and line.startswith("-"):
                evidence[current_section].append(line[1:].strip())
                
        return evidence

    def _extract_confidence(self, confidence_text: str) -> float:
        """Extract confidence percentage from text"""
        import re
        confidence_match = re.search(r"(\d+)%", confidence_text)
        if confidence_match:
            return float(confidence_match.group(1)) / 100
        return 0.5  # default confidence

    def _parse_response(self, response: str) -> WashTradingAnalysis:
        """Parse the GPT-4 response into structured analysis"""
        sections = {
            'reasoning_process': {},
            'evidence': {},
            'alternative_explanations': [],
            'is_wash_trading': False,
            'confidence': 0.0,
            'suspicious_addresses': [],
            'recommendations': [],
            'raw_response': response
        }

        try:
            # Parse reasoning process
            if "DETAILED REASONING:" in response:
                reasoning_text = response.split("DETAILED REASONING:")[1].split("EVIDENCE SUMMARY:")[0]
                sections['reasoning_process'] = {
                    'network': self._extract_section(reasoning_text, "Network Analysis"),
                    'volume': self._extract_section(reasoning_text, "Volume Analysis"),
                    'temporal': self._extract_section(reasoning_text, "Temporal Pattern Analysis"),
                    'price': self._extract_section(reasoning_text, "Price Pattern Analysis"),
                    'behavioral': self._extract_section(reasoning_text, "Behavioral Analysis")
                }

            # Parse evidence
            if "EVIDENCE SUMMARY:" in response:
                evidence_text = response.split("EVIDENCE SUMMARY:")[1].split("ALTERNATIVE EXPLANATIONS:")[0]
                sections['evidence'] = self._parse_evidence_sections(evidence_text)

            # Parse alternative explanations
            if "ALTERNATIVE EXPLANATIONS:" in response:
                alt_text = response.split("ALTERNATIVE EXPLANATIONS:")[1].split("WASH TRADING ASSESSMENT:")[0]
                sections['alternative_explanations'] = [exp.strip() for exp in alt_text.split("-") if exp.strip()]

            # Parse wash trading assessment
            if "WASH TRADING ASSESSMENT:" in response:
                assessment = response.split("WASH TRADING ASSESSMENT:")[1].split("CONFIDENCE LEVEL:")[0]
                sections['is_wash_trading'] = 'likely' in assessment.lower() or 'confirmed' in assessment.lower()

            # Parse confidence
            if "CONFIDENCE LEVEL:" in response:
                confidence_text = response.split("CONFIDENCE LEVEL:")[1].split("SUSPICIOUS ADDRESSES:")[0]
                sections['confidence'] = self._extract_confidence(confidence_text)

            if "SUSPICIOUS ADDRESSES:" in response:
                addresses_text = response.split("SUSPICIOUS ADDRESSES:")[1].split("\n")
                for line in addresses_text:
                    # Look for both backquoted addresses and plain addresses
                    if '`r' in line:
                        # Extract from backquotes
                        addr = line.split('`')[1]
                        if addr.startswith('r'):
                            sections['suspicious_addresses'].append(addr)
                    elif 'r' in line:
                        # Extract plain address
                        words = line.split()
                        for word in words:
                            if word.startswith('r') and len(word) >= 30:  # Minimum length check for address
                                # Clean up any punctuation
                                addr = word.strip('`-., \n')
                                sections['suspicious_addresses'].append(addr)

            # Parse recommendations
            if "RECOMMENDATIONS:" in response:
                recommendations_text = response.split("RECOMMENDATIONS:")[1]
                sections['recommendations'] = [
                    rec.strip() for rec in recommendations_text.split("-") 
                    if rec.strip()
                ]
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Return default structure if parsing fails
            
        return WashTradingAnalysis(**sections)

    def analyze(self, transactions: List[Dict]) -> WashTradingAnalysis:
        """Analyze transactions for wash trading patterns"""

        augmented_features = compute_all_metrics(transactions)
        # prompt = create_analysis_prompt(transactions)
        prompt = create_analysis_prompt(transactions, augmented_features)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            analysis = self._parse_response(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

def analyze_wash_trading(transactions: List[Dict], api_key: str) -> WashTradingAnalysis:
    """Convenience function for wash trading analysis"""
    detector = WashTradeDetector(api_key)
    return detector.analyze(transactions)
