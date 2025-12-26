import os
import datetime
from jinja2 import Environment, FileSystemLoader

class ReportGenerator:
    def __init__(self, template_dir=".", template_name="report_template.html"):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template_name = template_name

    def generate(self, data, output_path="report.html"):
        template = self.env.get_template(self.template_name)
        
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        html_content = template.render(**data)
        
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return os.path.abspath(output_path)
