def string_to_float_array(input_string):
    # 用逗号分割字符串，并将每个部分转化为浮点数
    float_array = [float(x) for x in input_string.split(',')]
    return float_array