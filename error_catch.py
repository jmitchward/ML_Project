@staticmethod
def menu_input(input_text, return_type):
    input_request = input(input_text)
    if return_type == 'int':
        int_input(input_request)
    else:
        string_input(input_request)


@staticmethod
def int_input(input_request):
    for each in range(len(input_request)):
        if input_request[each].isdigit():
            continue
        else:
            input_request = input_request.replace(input_request[each], '')
    return input_request


@staticmethod
def string_input(input_request):
    for each in range(len(input_request)):
        if input_request[each].isalpha():
            continue
        elif input_request == ' ':
            continue
        else:
            input_request = input_request.replace(input_request[each], '')
    return input_request.lower()
