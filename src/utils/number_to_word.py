import json
from itertools import chain


def number_to_word(num_str):
    numbers = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
    11: 'eleven',
    12: 'twelve',
    13: 'thirteen',
    14: 'fourteen',
    15: 'fifteen',
    16: 'sixteen',
    17: 'seventeen',
    18: 'eighteen',
    19: 'nineteen',
    20: 'twenty',
    30: 'thirty',
    40: 'forty',
    50: 'fifty',
    60: 'sixty',
    70: 'seventy',
    80: 'eighty',
    90: 'ninety'
    }
    # split with '.'
    if '.' in num_str:
        num = float(num_str)
    else:
        num = int(num_str)
    if '.' in str(num):
        whole, fraction = str(num).split('.')
    else:
        whole = str(num)
        fraction = None

    # int part
    whole_part = ''
    if len(whole) == 1:
        whole_part = numbers[int(whole)]
    elif len(whole) == 2:
        if whole.startswith('0'):
            whole_part = numbers[int(whole[1])]
        elif whole.startswith('1'):
            whole_part = numbers[int(whole)]
        else:
            whole_part = numbers[int(whole[0] + '0')] + ' ' + numbers[int(whole[1])]
    elif len(whole) == 3:
        whole_part = numbers[int(whole[0])] + ' hundred'
        if whole[1:] != '00':
            whole_part += ' and ' + number_to_word(whole[1:])
    elif len(whole) == 4:
        whole_part = numbers[int(whole[0])] + ' thousand'
        if whole[1:] != '000':
            whole_part += ' ' + number_to_word(whole[1:])

    # fraction part
    if fraction:
        fraction_part = 'point ' + ' '.join(numbers[int(digit)] for digit in fraction)
    else:
        fraction_part = ''
    
    return whole_part + ' ' + fraction_part
