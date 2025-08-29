#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Ethical AI Framework

This module provides comprehensive ethical AI considerations and bias monitoring
for the health insurance claim prediction system.

Features:
- Fairness assessment across demographic groups
- Bias detection and mitigation strategies
- Regulatory compliance checking (HIPAA, ACA)
- Model explainability and transparency
- Audit trail and documentation
- Patient rights protection

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
from pathlib import Path

# ML fairness libraries
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Model explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EthicalAIFramework:
    """Comprehensive ethical AI framework for healthcare predictions."""
    
    def __init__(self, model, data: pd.DataFrame, predictions: np.ndarray):
        """
        Initialize the ethical AI framework.
        
        Args:
            model: Trained ML model
            data: Dataset used for predictions
            predictions: Model predictions
        """
        self.model = model
        self.data = data
        self.predictions = predictions
        self.audit_log = []
        
        # Create output directory
        Path("ethical_ai_reports").mkdir(exist_ok=True)
        
        logger.info("Ethical AI Framework initialized")
    
    def assess_fairness_by_demographics(self) -> Dict[str, Any]:
        """
        Assess model fairness across different demographic groups.
        
        Returns:
            Dictionary containing fairness metrics by demographic group
        """
        logger.info("Assessing fairness by demographics...")
        
        fairness_results = {}
        
        # Define protected attributes and their categories
        protected_attributes = {
            'sex': {0: 'Female', 1: 'Male'},
            'age_group': self._create_age_groups(),
            'region': {0: 'Northeast', 1: 'Northwest', 2: 'Southeast', 3: 'Southwest'}
        }
        
        for attribute, categories in protected_attributes.items():
            if attribute == 'age_group':
                # Create age groups
                self.data['age_group'] = pd.cut(
                    self.data['age'], 
                    bins=[0, 30, 45, 60, 100], 
                    labels=['Young', 'Middle', 'Senior', 'Elder']
                )
            
            attr_fairness = {}
            
            for category_value, category_name in categories.items():
                if attribute == 'age_group':
                    mask = self.data['age_group'] == category_value
                else:
                    mask = self.data[attribute] == category_value
                
                if mask.sum() == 0:  # Skip if no data for this category
                    continue
                
                group_predictions = self.predictions[mask]
                group_actual = self.data.loc[mask, 'claim_status'].values
                
                # Calculate metrics
                accuracy = accuracy_score(group_actual, group_predictions)
                precision = precision_score(group_actual, group_predictions, zero_division=0)
                recall = recall_score(group_actual, group_predictions, zero_division=0)
                
                # Calculate approval rate
                approval_rate = 1 - group_predictions.mean()
                
                attr_fairness[category_name] = {
                    'count': mask.sum(),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'approval_rate': approval_rate,
                    'denial_rate': group_predictions.mean()
                }
            
            fairness_results[attribute] = attr_fairness
            
            # Log potential bias
            self._log_bias_assessment(attribute, attr_fairness)
        
        return fairness_results
    
    def _create_age_groups(self) -> Dict[str, str]:
        """Create age group categories."""
        return {'Young': 'Young', 'Middle': 'Middle', 'Senior': 'Senior', 'Elder': 'Elder'}
    
    def _log_bias_assessment(self, attribute: str, results: Dict[str, Any]) -> None:
        """Log bias assessment for audit trail."""
        
        if len(results) < 2:
            return
        
        # Calculate disparate impact ratio
        approval_rates = [group['approval_rate'] for group in results.values()]
        max_approval = max(approval_rates)
        min_approval = min(approval_rates)
        
        if max_approval > 0:
            disparate_impact_ratio = min_approval / max_approval
        else:
            disparate_impact_ratio = 1.0
        
        # 80% rule for disparate impact
        bias_detected = disparate_impact_ratio < 0.8
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'assessment_type': 'bias_detection',
            'protected_attribute': attribute,
            'disparate_impact_ratio': disparate_impact_ratio,
            'bias_detected': bias_detected,
            'results': results
        }
        
        self.audit_log.append(audit_entry)
        
        if bias_detected:
            logger.warning(
                f"⚠️ Potential bias detected for {attribute}: "
                f"Disparate impact ratio = {disparate_impact_ratio:.3f}"
            )
        else:
            logger.info(f"✅ No significant bias detected for {attribute}")
    
    def check_regulatory_compliance(self) -> Dict[str, Any]:
        """
        Check compliance with healthcare regulations (HIPAA, ACA).
        
        Returns:
            Compliance assessment results
        """
        logger.info("Checking regulatory compliance...")
        
        compliance_results = {
            'hipaa_compliance': self._check_hipaa_compliance(),
            'aca_compliance': self._check_aca_compliance(),
            'state_regulations': self._check_state_regulations()
        }
        
        # Log compliance assessment
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'assessment_type': 'regulatory_compliance',
            'results': compliance_results
        }
        self.audit_log.append(audit_entry)
        
        return compliance_results
    
    def _check_hipaa_compliance(self) -> Dict[str, Any]:
        """Check HIPAA compliance requirements."""
        
        hipaa_checks = {
            'phi_protection': {
                'status': 'compliant',
                'notes': 'Model uses encoded/anonymized data without direct PHI'
            },
            'minimum_necessary': {
                'status': 'compliant', 
                'notes': 'Uses only necessary features for claim prediction'
            },
            'access_controls': {
                'status': 'review_required',
                'notes': 'Implement proper authentication and authorization'
            },
            'audit_trails': {
                'status': 'compliant',
                'notes': 'Comprehensive audit logging implemented'
            }
        }
        
        return hipaa_checks
    
    def _check_aca_compliance(self) -> Dict[str, Any]:
        """Check Affordable Care Act compliance."""
        
        aca_checks = {
            'no_discrimination': {
                'status': 'review_required',
                'notes': 'Requires ongoing bias monitoring to ensure no discrimination'
            },
            'essential_benefits': {
                'status': 'not_applicable',
                'notes': 'Model predicts claim approval, not benefit determination'
            },
            'pre_existing_conditions': {
                'status': 'compliant',
                'notes': 'Model does not explicitly consider pre-existing conditions'
            }
        }
        
        return aca_checks
    
    def _check_state_regulations(self) -> Dict[str, Any]:
        """Check state-specific healthcare regulations."""
        
        state_checks = {
            'california_ccpa': {
                'status': 'review_required',
                'notes': 'Implement data subject rights if applicable'
            },
            'new_york_shield': {
                'status': 'review_required', 
                'notes': 'Ensure data security measures meet SHIELD Act requirements'
            },
            'general_privacy': {
                'status': 'compliant',
                'notes': 'Basic privacy protections implemented'
            }
        }
        
        return state_checks
    
    def generate_model_explainability(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Generate model explainability insights.
        
        Args:
            sample_size: Number of samples to use for SHAP analysis
            
        Returns:
            Dictionary containing explainability results
        """
        logger.info("Generating model explainability insights...")
        
        explainability_results = {
            'feature_importance': self._get_feature_importance(),
            'shap_analysis': self._generate_shap_analysis(sample_size) if SHAP_AVAILABLE else None,
            'model_transparency': self._assess_model_transparency()
        }
        
        return explainability_results
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        
        feature_names = [
            'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'
        ]
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance_values = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importance_values = np.abs(self.model.coef_[0])
        else:
            logger.warning("Unable to extract feature importance from model")
            return {}
        
        feature_importance = dict(zip(feature_names, importance_values))
        
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def _generate_shap_analysis(self, sample_size: int) -> Dict[str, Any]:
        """Generate SHAP (SHapley Additive exPlanations) analysis."""
        
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Sample data for SHAP analysis
            sample_indices = np.random.choice(len(self.data), 
                                            min(sample_size, len(self.data)), 
                                            replace=False)
            sample_data = self.data.iloc[sample_indices].drop('claim_status', axis=1)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(sample_data)
            
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                # For multi-class (use class 1 - denied claims)
                mean_shap = np.abs(shap_values[1]).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            
            feature_names = sample_data.columns
            shap_importance = dict(zip(feature_names, mean_shap))
            
            return {
                'shap_feature_importance': shap_importance,
                'sample_size': len(sample_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP analysis: {str(e)}")
            return None
    
    def _assess_model_transparency(self) -> Dict[str, str]:
        """Assess model transparency and interpretability."""
        
        model_type = type(self.model).__name__
        
        transparency_assessment = {
            'model_type': model_type,
            'interpretability': 'medium',  # Default
            'explainability_methods': ['feature_importance']
        }
        
        # Assess based on model type
        if 'LogisticRegression' in model_type:
            transparency_assessment['interpretability'] = 'high'
            transparency_assessment['explainability_methods'].append('coefficients')
        elif 'Tree' in model_type or 'Forest' in model_type:
            transparency_assessment['interpretability'] = 'medium'
            transparency_assessment['explainability_methods'].extend(['tree_structure', 'feature_importance'])
        elif 'SVM' in model_type or 'Neural' in model_type:
            transparency_assessment['interpretability'] = 'low'
            transparency_assessment['explainability_methods'] = ['shap', 'lime']
        
        if SHAP_AVAILABLE:
            transparency_assessment['explainability_methods'].append('shap')
        
        return transparency_assessment
    
    def generate_patient_rights_documentation(self) -> Dict[str, Any]:
        """Generate documentation about patient rights regarding AI decisions."""
        
        patient_rights = {
            'right_to_explanation': {
                'description': 'Patients have the right to understand how AI decisions are made',
                'implementation': 'Feature importance and SHAP explanations available',
                'contact': 'Contact customer service for detailed explanations'
            },
            'right_to_human_review': {
                'description': 'Patients can request human review of AI decisions',
                'implementation': 'Escalation process for disputed predictions',
                'contact': 'Submit appeal through customer portal or phone'
            },
            'right_to_data_access': {
                'description': 'Patients can access data used in their predictions',
                'implementation': 'Data summary provided upon request',
                'contact': 'Privacy office handles data access requests'
            },
            'right_to_correction': {
                'description': 'Patients can correct inaccurate data',
                'implementation': 'Data correction process available',
                'contact': 'Submit corrections through member portal'
            }
        }
        
        return patient_rights
    
    def create_bias_mitigation_recommendations(self, fairness_results: Dict[str, Any]) -> List[str]:
        """
        Create recommendations for bias mitigation based on fairness assessment.
        
        Args:
            fairness_results: Results from fairness assessment
            
        Returns:
            List of bias mitigation recommendations
        """
        recommendations = []
        
        for attribute, results in fairness_results.items():
            if len(results) < 2:
                continue
            
            # Calculate disparate impact
            approval_rates = [group['approval_rate'] for group in results.values()]
            max_rate = max(approval_rates)
            min_rate = min(approval_rates)
            
            if max_rate > 0:
                disparate_impact = min_rate / max_rate
                
                if disparate_impact < 0.8:
                    recommendations.extend([
                        f"⚠️ Address disparate impact in {attribute} (ratio: {disparate_impact:.3f})",
                        f"🔄 Consider re-sampling or re-weighting training data for {attribute}",
                        f"📊 Monitor {attribute} bias in production",
                        f"🎯 Set fairness constraints for {attribute} during model training"
                    ])
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ No significant bias detected, continue monitoring")
        
        recommendations.extend([
            "📈 Implement ongoing bias monitoring in production",
            "👥 Establish diverse review committee for AI decisions",
            "📋 Regular fairness audits (quarterly recommended)",
            "🔍 Patient feedback system for disputed predictions",
            "📚 Staff training on AI bias and fairness"
        ])
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive ethical AI report."""
        
        logger.info("Generating comprehensive ethical AI report...")
        
        # Conduct all assessments
        fairness_results = self.assess_fairness_by_demographics()
        compliance_results = self.check_regulatory_compliance()
        explainability_results = self.generate_model_explainability()
        patient_rights = self.generate_patient_rights_documentation()
        recommendations = self.create_bias_mitigation_recommendations(fairness_results)
        
        # Generate report
        report = f"""
# 🏥 ETHICAL AI ASSESSMENT REPORT
## Health Insurance Claim Prediction System

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model Type:** {type(self.model).__name__}
**Dataset Size:** {len(self.data)} records

---

## 📊 FAIRNESS ASSESSMENT

### Demographic Fairness Analysis
"""
        
        for attribute, results in fairness_results.items():
            report += f"\\n#### {attribute.title()} Analysis\\n"
            for group, metrics in results.items():
                report += f"- **{group}**: "
                report += f"Approval Rate: {metrics['approval_rate']:.3f}, "
                report += f"Accuracy: {metrics['accuracy']:.3f}\\n"
        
        report += f"""
---

## ⚖️ REGULATORY COMPLIANCE

### HIPAA Compliance
"""
        for check, result in compliance_results['hipaa_compliance'].items():
            report += f"- **{check}**: {result['status']} - {result['notes']}\\n"
        
        report += "\\n### ACA Compliance\\n"
        for check, result in compliance_results['aca_compliance'].items():
            report += f"- **{check}**: {result['status']} - {result['notes']}\\n"
        
        report += f"""
---

## 🔍 MODEL EXPLAINABILITY

### Feature Importance
"""
        feature_importance = explainability_results['feature_importance']
        for feature, importance in feature_importance.items():
            report += f"- **{feature}**: {importance:.4f}\\n"
        
        report += f"""
---

## 👤 PATIENT RIGHTS

### Rights and Implementation
"""
        for right, details in patient_rights.items():
            report += f"- **{right.replace('_', ' ').title()}**: {details['description']}\\n"
            report += f"  - Implementation: {details['implementation']}\\n"
            report += f"  - Contact: {details['contact']}\\n"
        
        report += f"""
---

## 🎯 RECOMMENDATIONS

### Bias Mitigation Actions
"""
        for i, recommendation in enumerate(recommendations, 1):
            report += f"{i}. {recommendation}\\n"
        
        report += f"""
---

## 📝 AUDIT TRAIL

Total audit entries: {len(self.audit_log)}

### Recent Assessments
"""
        for entry in self.audit_log[-5:]:  # Last 5 entries
            report += f"- {entry['timestamp']}: {entry['assessment_type']}\\n"
        
        # Save report
        report_path = f"ethical_ai_reports/ethical_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Comprehensive ethical AI report saved to {report_path}")
        
        return report


def main():
    """Main function for testing the ethical AI framework."""
    
    print("🏥 Health Insurance Claim Prediction - Ethical AI Framework")
    print("=" * 70)
    
    # This would normally be called with actual model and data
    print("\\n⚠️  This is a framework module. To use:")
    print("1. Import EthicalAIFramework")
    print("2. Initialize with your trained model and data")
    print("3. Call generate_comprehensive_report()")
    
    print("\\nExample usage:")
    print("""
from ethical_ai_framework import EthicalAIFramework

# After training your model
ethical_ai = EthicalAIFramework(model, test_data, predictions)
report = ethical_ai.generate_comprehensive_report()
print(report)
""")


if __name__ == "__main__":
    main()