"""
Pashto Education Prompt Engineering
Specialized prompts for educational content in Pashto
"""

class PashtoEducationPrompts:
    def __init__(self):
        self.educational_context = """تاسو د پښتو ژبې د زده کړې لپاره یو ګټور ښوونکی یاست. 
د افغانستان د کلتور او اسلامي ارزښتونو سره سم ځواب ورکړئ."""
    
    def format_educational_prompt(self, question: str) -> str:
        """Format question for educational context"""
        
        # Detect subject type
        if any(keyword in question.lower() for keyword in ["ریاضی", "حساب", "شمیرپوهنه"]):
            subject = "ریاضیاتو"
        elif any(keyword in question.lower() for keyword in ["ساینس", "علوم", "فزیک", "کیمیا"]):
            subject = "ساینسونو"
        elif any(keyword in question.lower() for keyword in ["تاریخ", "د افغانستان", "پخوانی"]):
            subject = "تاریخ"
        elif any(keyword in question.lower() for keyword in ["ژبه", "ګرامر", "املا"]):
            subject = "ژبې"
        elif any(keyword in question.lower() for keyword in ["دین", "اسلام", "قرآن"]):
            subject = "اسلامي زده کړو"
        else:
            subject = "عمومي پوهې"
        
        prompt = f"""{self.educational_context}

### د {subject} پوښتنه:
{question}

### د ښوونکي ځواب:
"""
        return prompt
    
    def get_subject_prompts(self, subject: str) -> dict:
        """Get subject-specific prompt templates"""
        
        prompts = {
            "math": {
                "context": "د ریاضیاتو ښوونکي په توګه، د دې مسئلې حل یې مرحله پر مرحله وښایاست:",
                "format": "مسئله: {question}\nحل:"
            },
            "science": {
                "context": "د ساینسونو ښوونکي په توګه، دا علمي پوښتنه ځوابولاست:",
                "format": "پوښتنه: {question}\nعلمي تشریح:"
            },
            "history": {
                "context": "د افغانستان د تاریخ ښوونکي په توګه، د دې تاریخي پوښتنې ځواب ورکړئ:",
                "format": "تاریخي پوښتنه: {question}\nتاریخي ځواب:"
            },
            "language": {
                "context": "د پښتو ژبې ښوونکي په توګه، د ژبې دا پوښتنه ځوابولاست:",
                "format": "ژبنۍ پوښتنه: {question}\nګرامري تشریح:"
            },
            "islamic": {
                "context": "د اسلامي زده کړو ښوونکي په توګه، د دې دیني پوښتنې ځواب ورکړئ:",
                "format": "دیني پوښتنه: {question}\nاسلامي ځواب:"
            }
        }
        
        return prompts.get(subject, prompts["math"])
    
    def create_interactive_prompt(self, student_level: str = "منځنۍ") -> str:
        """Create interactive learning prompt"""
        
        levels = {
            "لومړنۍ": "د لومړي ټولګي زده کوونکي لپاره",
            "منځنۍ": "د منځنۍ کچې زده کوونکي لپاره", 
            "لوړه": "د لیسې زده کوونکي لپاره",
            "پوهنتون": "د پوهنتون زده کوونکي لپاره"
        }
        
        level_context = levels.get(student_level, levels["منځنۍ"])
        
        return f"""تاسو یو تجربه کار او صبور ښوونکی یاست. {level_context} درس ورکوئ.

د ښه ښوونکي ځانګړتیاوې:
- ساده او روښانه تشریح
- د مثالونو کارول
- د زده کوونکي د پوښتنو هڅول
- د افغاني کلتور درناوی
- اسلامي ارزښتونو ته پاملرنه

اوس د زده کوونکي پوښتنه واورئ:"""