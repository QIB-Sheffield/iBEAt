import utilities.standardize_subject_name as standardize_subject_name

def standardize_name(subject_name):
    # Assuming standardize_subject_name is a module with the following functions:
    # subject_hifen, subject_underscore, subject_seven_digitd
    
    # Check if the subject_name is in the desired format
    if len(subject_name) != 7 or subject_name[4] != "_":
        # Try to standardize the name using different methods
        correct_name = standardize_subject_name.subject_hifen(subject_name)
        if correct_name != 0:
            if "followup" in subject_name:
                correct_name = correct_name + "_followup"
                return correct_name
            else:
                return correct_name 
        
        # Try the underscore method
        correct_name = standardize_subject_name.subject_underscore(subject_name)
        if correct_name != 0:
            if "followup" in subject_name:
                correct_name = correct_name + "_followup"
                return correct_name
            else:
                return correct_name 
        
        # Try the seven digits method
        correct_name = standardize_subject_name.subject_seven_digitd(subject_name)
        if correct_name != 0:
            if "followup" in subject_name:
                correct_name = correct_name + "_followup"
                return correct_name
            else:
                return correct_name 
    
    return None  # If none of the methods worked, return None