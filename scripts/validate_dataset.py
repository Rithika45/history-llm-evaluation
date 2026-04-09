print("QA_ID\tNo. of Choices\tNo. of Splits")

file_path = "C:/Users/Haojie/Desktop/llm-history-qa/history_qa_full.csv"
with open(file_path, 'r') as file_object:
	for line in file_object:
		data_fields = line.strip().split("|")
		if data_fields[0] != "Template_ID":
			#print(data_fields)
			template_id = data_fields[0]
			qa_template = data_fields[1]
			qa_id = data_fields[2]
			num_placeholders = int(data_fields[3])
			num_choices = int(data_fields[4])
			ground_truth = data_fields[5]
			# Parse all MCQ choices
			all_choices = data_fields[6].strip().split(";")
			for i in range(len(all_choices)):
				all_choices[i] = all_choices[i].strip()
			# Get the number of '[p' in template
			num_replacement = qa_template.count("[p")
			# Print out Question_ID, num_placeholders, num_replacement, Y/N
			# num_choices, len(all_choices), Y/N
			if num_placeholders == num_replacement:
				test_placeholders = "Y"
			else:
				test_placeholders = "N"

			if num_choices == len(all_choices):
				test_choices = "Y"
			else:
				test_choices = "N"

			if num_placeholders != num_replacement or num_choices != len(all_choices):
				print("{0}\t{1}\t\t{2}".format(qa_id, num_choices, len(all_choices)))
