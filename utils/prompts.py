from typing import Tuple


def build_task1_prompt(query: str, field: str) -> str:
    """
    Build a prompt template for single-choice questions.
    
    Args:
        query: The question content
        field: Professional field (e.g., physics, mathematics)
    
    Returns:
        Formatted prompt string
    """
    return "以下是关于{}的单项选择题，请选出正确答案并将选项填写到\\boxed{{}}中。\n\n{}".format(field, query)


def build_task2_prompt(query: str, text: str) -> str:
    """
    Build a prompt template for reading comprehension questions.
    
    Args:
        query: The question content
        text: Reading material
    
    Returns:
        Formatted prompt string
    """
    return (
        "阅读以下材料并回答问题，选出唯一正确答案并将选项填写到\\boxed{{}}中。\n\n"
        "# 材料：\n{}\n\n"
        "# 问题：\n{}"
    ).format(text, query)


def build_teacher_prompt(teaching_guide: str, text: str) -> Tuple[str, str]:
    """
    Build a prompt template for Chinese language teaching.
    
    Args:
        teaching_guide: Teaching guidance content
        text: Teaching material
    
    Returns:
        tuple: (System role prompt, Teaching prompt)
    """
    system_prompt = "你是一名国际汉语教师。"
    prompt = (
        "阅读以下材料，依据材料结合教学提示向学生传授相关知识，以{{\"knowledge\": 知识内容}}的格式输出。\n\n"
        "# 材料：\n{}\n\n"
        "# 教学提示：\n{}"
    ).format(text, teaching_guide)
    return system_prompt, prompt


def build_student_prompt(query: str, text: str) -> Tuple[str, str]:
    """
    Build a basic study prompt template for students.
    
    Args:
        query: The question content
        text: Learning material
    
    Returns:
        tuple: (System role prompt, Study prompt)
    """
    system_prompt = "你是一名正在学习汉语知识的学生。"
    prompt = (
        "阅读以下材料，选出唯一正确答案并将选项填写到\\boxed{{}}中。\n\n"
        "# 材料：\n{}\n\n"
        "# 问题：\n{}"
    ).format(text, query)
    return system_prompt, prompt


def build_guided_student_prompt(query: str, text: str, knowledge: str) -> Tuple[str, str]:
    """
    Build a study prompt template with teacher guidance.
    
    Args:
        query: The question content
        text: Learning material
        knowledge: Knowledge imparted by the teacher
    
    Returns:
        tuple: (System role prompt, Study prompt)
    """
    system_prompt = "你是一名正在学习汉语知识的学生。"
    prompt = (
        "阅读以下材料，结合教师传授的知识回答问题，选出唯一正确答案并将选项填写到\\boxed{{}}中。\n\n"
        "# 材料：\n{}\n\n"
        "# 教师传授的知识：\n{}\n\n"
        "# 问题：\n{}"
    ).format(text, knowledge, query)
    return system_prompt, prompt

if __name__ == "__main__":
    item={
        "text": '''
赌：由“赌”构成的词语，《大纲》中只有“赌博”一词，为六级词。  

·熟读下列句子，体会画线词语的意思。  

$\textcircled{1}$ 她的丈夫是个<u>赌棍</u> —一个专靠赌博吃饭的人。  
$\textcircled{2}$ 他们径直朝那幢高大的白色大楼走去，那里尽是渴望赚钱的<u>赌徒</u>。  
$\textcircled{3}$ 妈妈苦口婆心地嘱咐他：“以后可不要再<u>赌</u>了，要知道赌钱不是正经人所为。”  
$\textcircled{4}$ 生活对于雄心勃勃的人是一场精彩绝伦的<u>赌博</u>，需要他用尽全部的智慧、精力和勇气。  
''',
        "question":'''
·为下列句子中的“打赌”选择意思最接近的解释。  

$\textcircled{1}$ 我用脑袋<u>打赌</u>，你说谎！
$\textcircled{2}$ 她没有发疯，也不是傻子，她是真诚的。我可以用我的生命<u>打赌</u>。  

A. 为了贏，舍得花大钱
B. 因为怕输，不敢下赌注
C. 与别人对赌谁输谁赢  
''',
        "answer": "C"
    }
    system, question = build_student_prompt(item["question"], item["text"])
    print(system)
    print(question)
