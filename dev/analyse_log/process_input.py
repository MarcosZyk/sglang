import csv

def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header for the new CSV file
        writer.writerow(['timestamp', 'seq_id', 'agent_id', 'task_id', 'page_id', 'event', 'status'])
        
        current_agent_id = None
        current_task_id = None
        
        for row in reader:
            if len(row) == 0:
                continue
            
            # Check if this is a title record by looking at the first column
            if row[0].startswith('seq_id'):
                # This is a title record, extract agent_id and task_id
                headers = row
                values = next(reader)  # The next line contains the actual values
                
                # Create a dictionary from headers and values
                data_dict = dict(zip(headers, values))
                
                current_agent_id = data_dict.get('agent_id')
                current_task_id = data_dict.get('task_id')
            else:
                # This is an event record, insert agent_id and task_id
                timestamp, seq_id, page_id, event, status = row[:5]
                writer.writerow([timestamp, seq_id, page_id, current_agent_id, current_task_id, event, status])

if __name__ == "__main__":
    input_filename = "input.csv"
    output_filename = "output.csv"
    
    process_csv(input_filename, output_filename)
    print(f"Processed {input_filename} -> {output_filename}")