"""
Check and Analyze HF Hub Models for ZamAI
This script will:
1. List all models in your HF Hub
2. Check their status and compatibility
3. Prepare them for fine-tuning/training
4. Validate model configurations
"""

import json
import os
from typing import Dict, List, Optional
from huggingface_hub import HfApi, ModelCard, list_models
from huggingface_hub.utils import RepositoryNotFoundError
import pandas as pd
from datetime import datetime

class ZamAIHubModelAnalyzer:
    """Analyze and manage models in HF Hub for ZamAI"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with HF token"""
        if token is None:
            try:
                with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
                    token = f.read().strip()
            except FileNotFoundError:
                raise ValueError("HF token required. Place it in HF-Token.txt")
        
        self.api = HfApi(token=token)
        self.token = token
        self.username = self._get_username()
    
    def _get_username(self) -> str:
        """Get the username from the token"""
        try:
            user_info = self.api.whoami()
            return user_info['name']
        except Exception as e:
            print(f"Could not get username: {e}")
            return "unknown"
    
    def list_my_models(self) -> List[Dict]:
        """List all models owned by the user"""
        print(f"🔍 Checking models for user: {self.username}")
        
        try:
            models = list(self.api.list_models(author=self.username))
            model_info = []
            
            for model in models:
                info = {
                    'id': model.id,
                    'private': model.private,
                    'downloads': model.downloads,
                    'likes': model.likes,
                    'created_at': model.created_at.isoformat() if model.created_at else None,
                    'last_modified': model.last_modified.isoformat() if model.last_modified else None,
                    'tags': model.tags,
                    'pipeline_tag': model.pipeline_tag,
                    'library_name': model.library_name
                }
                model_info.append(info)
            
            print(f"✅ Found {len(model_info)} models")
            return model_info
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def get_model_details(self, model_id: str) -> Dict:
        """Get detailed information about a specific model"""
        try:
            # Get model info
            model_info = self.api.model_info(model_id)
            
            # Try to get model card
            try:
                card = ModelCard.load(model_id)
                card_content = card.content if hasattr(card, 'content') else str(card)
            except:
                card_content = "No model card available"
            
            # Get files in the repository
            try:
                files = self.api.list_repo_files(model_id)
            except:
                files = []
            
            details = {
                'id': model_id,
                'sha': model_info.sha,
                'private': model_info.private,
                'downloads': model_info.downloads,
                'likes': model_info.likes,
                'tags': model_info.tags,
                'pipeline_tag': model_info.pipeline_tag,
                'library_name': model_info.library_name,
                'created_at': model_info.created_at.isoformat() if model_info.created_at else None,
                'last_modified': model_info.last_modified.isoformat() if model_info.last_modified else None,
                'model_card': card_content,
                'files': files,
                'siblings': [{'filename': s.rfilename, 'size': getattr(s, 'size', 0)} for s in model_info.siblings]
            }
            
            return details
            
        except RepositoryNotFoundError:
            print(f"❌ Model {model_id} not found")
            return {}
        except Exception as e:
            print(f"❌ Error getting details for {model_id}: {e}")
            return {}
    
    def analyze_model_for_training(self, model_id: str) -> Dict:
        """Analyze if model is ready for training/fine-tuning"""
        details = self.get_model_details(model_id)
        if not details:
            return {'ready': False, 'issues': ['Model not found']}
        
        analysis = {
            'ready': True,
            'issues': [],
            'recommendations': [],
            'training_compatible': False,
            'base_model_type': 'unknown',
            'has_tokenizer': False,
            'has_config': False,
            'estimated_size': 0
        }
        
        files = details.get('files', [])
        siblings = details.get('siblings', [])
        
        # Check for essential files
        essential_files = {
            'config.json': False,
            'pytorch_model.bin': False,
            'model.safetensors': False,
            'tokenizer.json': False,
            'tokenizer_config.json': False
        }
        
        for file_info in siblings:
            filename = file_info['filename']
            if filename in essential_files:
                essential_files[filename] = True
            if 'tokenizer' in filename.lower():
                analysis['has_tokenizer'] = True
            if filename == 'config.json':
                analysis['has_config'] = True
            
            # Estimate size
            analysis['estimated_size'] += file_info.get('size', 0)
        
        # Check model compatibility
        tags = details.get('tags', [])
        pipeline_tag = details.get('pipeline_tag', '')
        library_name = details.get('library_name', '')
        
        # Determine if suitable for training
        if 'transformers' in library_name or 'pytorch' in tags:
            analysis['training_compatible'] = True
        
        # Determine base model type
        if any(tag in tags for tag in ['llama', 'llama2', 'llama3']):
            analysis['base_model_type'] = 'llama'
        elif 'mistral' in tags or 'mixtral' in tags:
            analysis['base_model_type'] = 'mistral'
        elif 'phi' in tags:
            analysis['base_model_type'] = 'phi'
        elif 'bloom' in tags:
            analysis['base_model_type'] = 'bloom'
        elif pipeline_tag == 'text-generation':
            analysis['base_model_type'] = 'text-generation'
        
        # Check for issues
        if not (essential_files['config.json']):
            analysis['issues'].append('Missing config.json')
            analysis['ready'] = False
        
        if not (essential_files['pytorch_model.bin'] or essential_files['model.safetensors']):
            analysis['issues'].append('Missing model weights file')
            analysis['ready'] = False
        
        if not analysis['has_tokenizer']:
            analysis['issues'].append('Missing tokenizer files')
            analysis['ready'] = False
        
        # Recommendations
        if not analysis['training_compatible']:
            analysis['recommendations'].append('Ensure model uses transformers library')
        
        if analysis['estimated_size'] == 0:
            analysis['recommendations'].append('Check if model files are properly uploaded')
        
        if not details.get('model_card') or 'No model card' in details.get('model_card', ''):
            analysis['recommendations'].append('Add a comprehensive model card')
        
        return analysis
    
    def check_zamai_models(self) -> Dict:
        """Check specifically for ZamAI models based on naming patterns"""
        all_models = self.list_my_models()
        
        zamai_models = []
        other_models = []
        
        zamai_keywords = ['zamai', 'pashto', 'afghan', 'dari']
        
        for model in all_models:
            model_id = model['id'].lower()
            if any(keyword in model_id for keyword in zamai_keywords):
                zamai_models.append(model)
            else:
                other_models.append(model)
        
        return {
            'zamai_models': zamai_models,
            'other_models': other_models,
            'total_models': len(all_models),
            'zamai_count': len(zamai_models)
        }
    
    def prepare_training_config(self, model_id: str) -> Dict:
        """Generate training configuration for a model"""
        analysis = self.analyze_model_for_training(model_id)
        
        if not analysis['ready']:
            return {'error': 'Model not ready for training', 'issues': analysis['issues']}
        
        # Base configuration
        config = {
            'base_model': model_id,
            'model_type': analysis['base_model_type'],
            'training_ready': True,
            'recommended_settings': {}
        }
        
        # Model-specific recommendations
        if analysis['base_model_type'] == 'llama':
            config['recommended_settings'] = {
                'lora': {
                    'rank': 64,
                    'alpha': 128,
                    'dropout': 0.05,
                    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                },
                'training': {
                    'batch_size': 2,
                    'gradient_accumulation': 8,
                    'learning_rate': 2e-5,
                    'max_length': 2048
                }
            }
        elif analysis['base_model_type'] == 'mistral':
            config['recommended_settings'] = {
                'lora': {
                    'rank': 32,
                    'alpha': 64,
                    'dropout': 0.1,
                    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
                },
                'training': {
                    'batch_size': 4,
                    'gradient_accumulation': 4,
                    'learning_rate': 1e-5,
                    'max_length': 1024
                }
            }
        else:
            config['recommended_settings'] = {
                'lora': {
                    'rank': 16,
                    'alpha': 32,
                    'dropout': 0.1
                },
                'training': {
                    'batch_size': 4,
                    'gradient_accumulation': 4,
                    'learning_rate': 5e-6
                }
            }
        
        return config
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all models"""
        print("📊 Generating comprehensive model report...")
        
        models_data = self.check_zamai_models()
        all_models = models_data['zamai_models'] + models_data['other_models']
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'username': self.username,
            'summary': {
                'total_models': models_data['total_models'],
                'zamai_models': models_data['zamai_count'],
                'other_models': len(models_data['other_models'])
            },
            'models': []
        }
        
        for model in all_models:
            model_id = model['id']
            print(f"  📋 Analyzing {model_id}...")
            
            analysis = self.analyze_model_for_training(model_id)
            training_config = self.prepare_training_config(model_id)
            
            model_report = {
                'id': model_id,
                'basic_info': model,
                'training_analysis': analysis,
                'training_config': training_config,
                'is_zamai_model': any(keyword in model_id.lower() for keyword in ['zamai', 'pashto', 'afghan', 'dari'])
            }
            
            report['models'].append(model_report)
        
        # Save report
        report_path = '/workspaces/ZamAI-Pro-Models/data/processed/hf_hub_model_report.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Report saved to: {report_path}")
        return report_path

def main():
    """Main function to analyze HF Hub models"""
    print("🔍 ZamAI HF Hub Model Analyzer")
    print("=" * 50)
    
    try:
        analyzer = ZamAIHubModelAnalyzer()
        
        # Generate comprehensive report
        report_path = analyzer.generate_report()
        
        # Load and display summary
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print("\n📊 SUMMARY")
        print("-" * 30)
        print(f"Total Models: {report['summary']['total_models']}")
        print(f"ZamAI Models: {report['summary']['zamai_models']}")
        print(f"Other Models: {report['summary']['other_models']}")
        
        print("\n🎯 ZAMAI MODELS")
        print("-" * 30)
        for model in report['models']:
            if model['is_zamai_model']:
                model_id = model['id']
                ready = model['training_analysis']['ready']
                status = "✅ Ready" if ready else "❌ Issues"
                print(f"{model_id}: {status}")
                
                if not ready:
                    for issue in model['training_analysis']['issues']:
                        print(f"  - {issue}")
        
        print("\n🔧 TRAINING RECOMMENDATIONS")
        print("-" * 30)
        training_ready_count = 0
        for model in report['models']:
            if model['training_analysis']['ready']:
                training_ready_count += 1
                model_id = model['id']
                model_type = model['training_analysis']['base_model_type']
                print(f"✅ {model_id} ({model_type}) - Ready for training")
        
        print(f"\n🎉 {training_ready_count} models ready for training!")
        
        # Check for expected ZamAI models from configs
        print("\n🔍 CHECKING EXPECTED MODELS")
        print("-" * 30)
        
        config_path = '/workspaces/ZamAI-Pro-Models/configs/pashto_chat_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            expected_model = config.get('hub_model_id', '')
            if expected_model:
                found = any(model['id'] == expected_model for model in report['models'])
                if found:
                    print(f"✅ Expected model found: {expected_model}")
                else:
                    print(f"⚠️  Expected model not found: {expected_model}")
                    print("   This model may need to be trained and uploaded")
        
        return report
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
