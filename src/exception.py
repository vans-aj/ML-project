import sys

def error_messege_details(error,error_details: sys):
    """
    Returns a detailed error message based on the type of exception.
    """
    _,_, exc_tb = error_details.exc_info()
    error_message = f"Error occurred in script: [{exc_tb.tb_frame.f_code.co_filename}] at line number: [{exc_tb.tb_lineno}] with error message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_messege_details(error_message, error_details = error_details)
    
    def __str__(self):
        return self.error_message
    
