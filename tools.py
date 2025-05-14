import json
import os
from datetime import datetime
import re
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt

def validate_json(json_string):
    try:
        json.loads(json_string)  # Attempt to parse the JSON string
        return True
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return False
    
def save_judge_feedback(feedback: str, filename: str):
    print('saving feedback')
    try:
        with open(f"{filename}.md", "w") as file:
            file.write(feedback)
        print("printed")
    except Exception as e:
        return {"status": "error", "message": f"Failed to save table: {e}"}

def save_table(filename: str, table: str) -> None:
    # Validate the JSON format before proceeding
    if not validate_json(table):
        return {"status": "error", "message": "Invalid JSON format. Table not saved."}
    
    try:
        table = json.loads(table)  # Convert the string to a dictionary
        with open(filename, 'w') as file:
            json.dump(table, file, indent=4)

        return {"status": "success", "message": "Table saved successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save table: {e}"}
    
def save_domain_needs(table: Union[str, dict], result_file_path: str) -> dict:
    """
    Save domain needs to a JSON file.
    
    Args:
        table: Either a JSON string or a dictionary containing the domain needs
        result_file_path: Path where the file should be saved
        
    Returns:
        dict: Status of the operation
    """
    try:
        print("============save_domain_needs============")
        # If table is a string, parse it to a dictionary
        if isinstance(table, str):
            table = json.loads(table)
        # If table is already a dictionary, use it directly
        
        # Save to the result folder
        with open(result_file_path, 'w') as file:
            json.dump(table, file, indent=4)

        return {"status": "success", "message": "Table saved successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save table: {e}"}


def save_table_as_image(filename: str, data: dict) -> None:
    rows = []
    metric_no = 1
    need_no_dict = {need: idx + 1 for idx, need in enumerate(data.keys())}  # Assign unique numbers to needs

    for need, metrics in data.items():
        for metric, unit in metrics:
            # Check if the metric already exists in the rows
            existing_row = next((row for row in rows if row[2] == metric and row[3] == unit), None)
            if existing_row:
                # Append the need number to the existing row
                existing_row[1] = str(existing_row[1]) + f", {need_no_dict[need]}"
            else:
                # Create a new row for the metric
                rows.append([metric_no, need_no_dict[need], metric, unit])
                metric_no += 1

    df_metrics = pd.DataFrame(rows, columns=['Metric No.', 'Need No.', 'Metric Description', 'Units'])

    need_no_list=list(need_no_dict.items())
    need_no_list=[[j,i] for i,j in need_no_list]
    
    df_needs = pd.DataFrame(need_no_list, columns=['Need No.', 'Customer Need'])

    # Plotting the DataFrames as tables
    fig, ax = plt.subplots(figsize=(10, len(rows) // 2 + len(df_needs) // 2))
    ax.axis('off')

    # First table for metrics
    table_metrics = ax.table(
        cellText=df_metrics.values,
        colLabels=df_metrics.columns,
        cellLoc='center',
        loc='center'
    )
    table_metrics.auto_set_column_width(col=[0, 1, 2, 3])

    # Adjust the position of the second table to add a gap
    ax2 = fig.add_subplot(212)  # Add a new subplot for the second table
    ax2.axis('off')
    plt.subplots_adjust(hspace=0.5)  # Add space between the two tables

    table_needs = ax2.table(
        cellText=df_needs.values,
        colLabels=df_needs.columns,
        cellLoc='center',
        loc='center'
    )
    table_needs.auto_set_column_width(col=[0, 1])

    # Save the figure in the outcome directory
    plt.savefig(filename, bbox_inches='tight', dpi=300)

def extract_and_save_scores(response, result_folder: str) -> None:
    """
    Extract scores and issues from judge's response and save them to a file.
    Expected format: "Overall Score: X/5\nBreakdown:\n- Specificity: X/5\n- Relevance: X/5\n- Non-Redundancy: X/5\n3. Issues Found: ..."
    """
    try:
        # Convert response to string if it's a Response object
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
            
        print("Judge's response:", response_text)  # Debug print
        
        # Extract scores using more specific regex patterns
        # Pattern to match scores in the expected format
        overall_score = re.search(r'Overall Score:\s*(\d+(?:\.\d+)?)/5', response_text)
        specificity_score = re.search(r'Specificity:\s*(\d+(?:\.\d+)?)/5', response_text)
        relevance_score = re.search(r'Relevance:\s*(\d+(?:\.\d+)?)/5', response_text)
        non_redundancy_score = re.search(r'Non-Redundancy:\s*(\d+(?:\.\d+)?)/5', response_text)
        
        # Extract the scores if found
        scores = []
        for match in [overall_score, specificity_score, relevance_score, non_redundancy_score]:
            if match:
                scores.append(match.group(1))
            else:
                print(f"Warning: Could not find one of the expected scores in the response")
                return
        
        if len(scores) != 4:
            print(f"Warning: Could not extract all 4 scores from response. Found {len(scores)} scores: {scores}")
            return
            
        # Extract Issues Found section
        issues_match = re.search(r'3\. Issues Found:(.*?)(?=\n\n|\Z)', response_text, re.DOTALL)
        issues_text = issues_match.group(1).strip() if issues_match else ""
        
        # Split issues by category
        issues_dict = {}
        categories = ["Specificity", "Relevance", "Non-Redundancy"]
        for category in categories:
            category_match = re.search(f"{category}:(.*?)(?=\n\n|\Z)", issues_text, re.DOTALL)
            if category_match:
                issues = [issue.strip('- ').strip() for issue in category_match.group(1).split('\n') if issue.strip()]
                issues_dict[category.lower()] = issues
        
        # Create scores dictionary with float values and issues
        scores_dict = {
            "overall": float(scores[0]),
            "specificity": float(scores[1]),
            "relevance": float(scores[2]),
            "non_redundancy": float(scores[3]),
            "issues": issues_dict,
            "raw_response": response_text  # Store the full response for reference
        }
        
        # Save to file
        scores_file = os.path.join(result_folder, "judge_scores.json")
        with open(scores_file, 'w') as f:
            json.dump(scores_dict, f, indent=4)
        print(f"Scores and issues saved to {scores_file}")
        
    except Exception as e:
        print(f"Error extracting scores and issues: {e}")
        print(f"Response type: {type(response)}")
        print(f"Response content: {response}")

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = f"./results/{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    response =  '''
        1. Overall Score: 4.8/5

    2. Breakdown:
    - Specificity: 4/5
    - Relevance: 4.1/5
    - Non-Redundancy: 4/5

    3. Issues Found:
    - **Specificity:** M
'''
    extract_and_save_scores(response, result_folder)