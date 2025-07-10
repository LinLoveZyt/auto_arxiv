# utils/pdf_generator.py

import logging
import json
import subprocess
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List

# ▼▼▼ [修改] 改变导入方式 ▼▼▼
from core import config as config_module

logger = logging.getLogger(__name__)

def escape_latex_text(text: str) -> str:
    """对普通文本中的 LaTeX 特殊字符进行转义，以防止编译错误。"""
    if not isinstance(text, str):
        return str(text)
    
    parts = re.split(r'(\$\$?.*?\$\$?)', text)
    
    conv = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', 
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\^{}',
    }
    regex = re.compile('|'.join(re.escape(key) for key in conv.keys()))

    escaped_parts = []
    for i, part in enumerate(parts):
        if part.startswith('$') and part.endswith('$'):
            escaped_parts.append(part)
        else:
            escaped_parts.append(regex.sub(lambda match: conv[match.group(0)], part))
            
    return "".join(escaped_parts)


def format_paper_latex(paper_data: Dict[str, Any], language: str = 'en') -> str:
    """
    将单篇论文的JSON数据格式化为LaTeX代码块，包含图片、分类和收录理由。
    """
    
    title = escape_latex_text(paper_data.get('title', 'N/A'))
    arxiv_id = escape_latex_text(paper_data.get('arxiv_id', ''))
    published_date_raw = paper_data.get('published_date', '')
    published_date = str(published_date_raw).split('T')[0] if published_date_raw else ''
    
    classification = paper_data.get('classification', {})
    domain = escape_latex_text(classification.get('domain', 'N/A'))
    task = escape_latex_text(classification.get('task', 'N/A'))

    analysis = paper_data.get('analysis', {})
    problem = escape_latex_text(analysis.get('problem_solved', 'N/A'))
    originality = escape_latex_text(analysis.get('originality', 'N/A'))
    comparison = escape_latex_text(analysis.get('method_comparison', 'N/A'))
    
    # [核心修改] 获取收录理由
    justification = escape_latex_text(paper_data.get('selection_justification', ''))

    headers = {
        'en': {'problem': 'Problem Solved', 'originality': 'Originality & Innovation', 'comparison': 'Method Comparison', 'fig_arch': 'Figure: Model Architecture/Workflow', 'fig_perf': 'Figure: Performance Comparison', 'category': 'Category', 'reason': 'Reason for Inclusion'},
        'zh': {'problem': '解决的问题', 'originality': '独创性与创新点', 'comparison': '方法对比', 'fig_arch': '图：模型架构/工作流程', 'fig_perf': '图：性能对比', 'category': '分类', 'reason': '收录理由'}
    }
    lang_headers = headers.get(language, headers['en'])

    latex_str = f"\\section*{{{title}}}\n"
    latex_str += f"\\subsection*{{{lang_headers['category']}: {domain} / {task} --- arXiv: {arxiv_id} --- Published: {published_date}}}\n\\vspace{{-1em}}\\hrulefill\n\n"
    
    # [核心修改] 如果有收录理由，则将其添加到报告中
    if justification:
        latex_str += f"\\subsubsection*{{{lang_headers['reason']}}}\n\\textcolor{{blue}}{{{justification}}}\n\n"

    latex_str += f"\\subsubsection*{{{lang_headers['problem']}}}\n{problem}\n\n"
    latex_str += f"\\subsubsection*{{{lang_headers['originality']}}}\n{originality}\n\n"
    latex_str += f"\\subsubsection*{{{lang_headers['comparison']}}}\n{comparison}\n\n"

    images_info = paper_data.get('images', {})
    arch_img_path_str = images_info.get('architecture_image')
    perf_img_path_str = images_info.get('performance_image')

    base_image_dir = config_module.STRUCTURED_DATA_DIR / paper_data.get('arxiv_id', '').replace('/', '_')
    
    if arch_img_path_str:
        absolute_path = (base_image_dir / arch_img_path_str).resolve()
        if absolute_path.exists():
            latex_str += "\\begin{figure}[h!]\n\\centering\n"
            latex_str += f"\\includegraphics[width=0.8\\textwidth]{{{str(absolute_path)}}}\n"
            latex_str += f"\\caption{{{escape_latex_text(lang_headers['fig_arch'])}}}\n"
            latex_str += "\\end{figure}\n\\vspace{1em}\n"

    if perf_img_path_str:
        absolute_path = (base_image_dir / perf_img_path_str).resolve()
        if absolute_path.exists():
            latex_str += "\\begin{figure}[h!]\n\\centering\n"
            latex_str += f"\\includegraphics[width=0.8\\textwidth]{{{str(absolute_path)}}}\n"
            latex_str += f"\\caption{{{escape_latex_text(lang_headers['fig_perf'])}}}\n"
            latex_str += "\\end{figure}\n\\vspace{1em}\n"

    return latex_str


def generate_daily_report_pdf(report_data: Dict[str, Any], output_path: Path, language: str = 'en'):
    """
    Generates a LaTeX PDF based on the given report data dictionary, including a statistics table.
    """
    # ▼▼▼ [修改] 在函数内部获取配置 ▼▼▼
    current_config = config_module.get_current_config()
    report_author = escape_latex_text(current_config['REPORT_AUTHOR'])

    report_title = escape_latex_text(report_data.get('report_title', 'Daily arXiv Report'))
    report_date = escape_latex_text(report_data.get('report_date', ''))
    papers = report_data.get('papers', [])
    statistics = report_data.get('statistics', {})

    if not papers:
        logger.warning("No papers in the report data, skipping PDF generation.")
        return

    documentclass = r'\documentclass[UTF8,a4paper,11pt]{ctexart}' if language == 'zh' else r'\documentclass[a4paper,11pt]{article}'
    
    latex_header = f"""
{documentclass}
\\usepackage{{amsmath, amssymb, amsfonts}}
\\usepackage[left=2.5cm, right=2.5cm, top=2.5cm, bottom=2.5cm]{{geometry}}
\\usepackage{{fancyhdr}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{titling}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}

\\definecolor{{TitleBlue}}{{RGB}}{{0, 82, 155}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[C]{{}}
\\fancyfoot[C]{{\\thepage}}
\\renewcommand{{\\headrulewidth}}{{0.4pt}}
\\renewcommand{{\\footrulewidth}}{{0.4pt}}

\\pretitle{{\\begin{{center}}\\Huge\\bfseries\\color{{TitleBlue}}}}
\\posttitle{{\\end{{center}}}}
\\preauthor{{\\begin{{center}}\\large}}
\\postauthor{{\\end{{center}}}}
\\predate{{\\begin{{center}}\\large}}
\\postdate{{\\end{{center}}}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=TitleBlue,
    urlcolor=cyan,
    pdftitle={{{report_title}}},
    pdfauthor={{{report_author}}}
}}

\\linespread{{1.3}}

\\begin{{document}}
"""
    latex_footer = r"""
\end{document}
"""
    
    title_part = f"\\title{{{report_title}}}\n\\author{{{report_author}}}\n\\date{{{report_date}}}\n\\maketitle\n"
    
    stats_part = ""
    if statistics:
        total_papers = statistics.get("total_papers", 0)
        breakdown = statistics.get("breakdown", {})
        
        stats_headers = {'en': ('Summary of Today\'s Papers', 'Domain', 'Task', 'Count'), 'zh': ('今日论文总览', '领域', '任务', '数量')}
        lang_stats_headers = stats_headers.get(language, stats_headers['en'])

        stats_part += f"\\section*{{{lang_stats_headers[0]}: {total_papers}}}\n"
        stats_part += "\\begin{center}\n\\begin{tabular}{llr}\n"
        stats_part += f"\\toprule\n\\textbf{{{lang_stats_headers[1]}}} & \\textbf{{{lang_stats_headers[2]}}} & \\textbf{{{lang_stats_headers[3]}}} \\\\\n\\midrule\n"
        
        for domain, tasks in breakdown.items():
            domain_escaped = escape_latex_text(domain)
            is_first_in_domain = True
            for task, count in tasks.items():
                task_escaped = escape_latex_text(task)
                if is_first_in_domain:
                    stats_part += f"{domain_escaped} & {task_escaped} & {count} \\\\\n"
                    is_first_in_domain = False
                else:
                    stats_part += f" & {task_escaped} & {count} \\\\\n"
            stats_part += "\\midrule\n"

        stats_part += "\\bottomrule\n\\end{tabular}\n\\end{center}\n\\clearpage\n"

    content_parts = [latex_header, title_part, stats_part]
    
    for paper in papers:
        content_parts.append(format_paper_latex(paper, language=language))
        content_parts.append("\n\\clearpage\n")
    
    content_parts.append(latex_footer)
    full_latex_doc = "".join(content_parts)

    output_dir = output_path.parent
    base_filename = output_path.stem
    tex_filepath = output_dir / f"{base_filename}.tex"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(tex_filepath, "w", encoding="utf-8") as f:
            f.write(full_latex_doc)
        logger.info(f"Successfully generated LaTeX source file: {tex_filepath}")
    except IOError as e:
        logger.error(f"Failed to write LaTeX file '{tex_filepath}': {e}")
        return

    if not shutil.which("xelatex"):
        logger.error("Command 'xelatex' not found. Please ensure a full LaTeX distribution is installed. PDF generation is skipped.")
        return

    # ▼▼▼ [核心修改] 引入编译成功标志位 ▼▼▼
    compilation_successful = True
    for i in range(2):
        logger.info(f"--- Starting LaTeX compilation pass {i+1}/2 ---")
        try:
            process = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", tex_filepath.name],
                cwd=output_dir,
                capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            # 只在最后一次编译后检查返回码
            if i == 1 and process.returncode != 0:
                logger.error(f"LaTeX compilation failed after 2 passes. See log file for details: {output_dir / (base_filename + '.log')}")
                compilation_successful = False # 标记编译失败
        except Exception as e:
            logger.critical(f"A critical error occurred during LaTeX compilation: {e}")
            compilation_successful = False
            break # 出现严重异常，直接跳出循环
    
    final_pdf_path = output_dir / f"{base_filename}.pdf"
    
    # ▼▼▼ [核心修改] 仅在编译成功且文件存在时，才宣告成功并清理文件 ▼▼▼
    if final_pdf_path.exists() and compilation_successful:
        logger.info(f"🎉 Successfully generated PDF report: {final_pdf_path}")
        for ext in ['.aux', '.log', '.out', '.toc', '.tex']:
            aux_file = output_dir / f"{base_filename}{ext}"
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except OSError:
                    pass
    else:
        logger.error(f"❌ PDF file generation failed. Please check the '{base_filename}.log' file in '{output_dir}' for detailed error messages.")