
import os
import math
import pandas as pd
import numpy as np
from statistics import harmonic_mean


from IPython.display import display, Markdown, Latex

#   <  bold  >
#   <  underline  >
#   <  strike  >
#   <  hr (HUMAN READABLE)  >
#   <  percent  >

def get_formating():
    print(
        '''
from formating import bold, underline, strike, hr, percent
        '''
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  col_cap  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
def col_cap(my_string):
    '''
    Takes in the name of a column. Assumes that the column name is all lower case and split_with_underscores.
    
    returns the column name Split With Underscores (lol, not with underscores and each word is capitalized)
    
    usually used to make a title for a plot from a column name
    
    Parameters:
    --------------
    my_string : a string
    
    Returns:
    --------------
    a better string (if you want to use it in a title)
    '''
    # new_string = my_string.split('_')

    # j=''
    # for i in new_string:
    #     j += i.capitalize()
    #     j += ' '

    # j = j.strip()

    new_string = ' '.join([x.capitalize() for x in my_string.split('_')])
    return new_string
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  BOLD  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def bold(text):
    result = '\033[1m' + text + '\033[0m'
    return result


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  UNDERLINE  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def underline(text):
    result = ''
    for c in text:
        result = result + c + '\u0332'
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  STRIKE  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  HR (HUMAN READABLE)  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def hr(n, suffix='', places=2, prefix='$'):
    '''
    Return a human friendly approximation of n, using SI prefixes

    '''
    prefixes = ['','K','M','B','T']
    
    # if n <= 99_999:
    #     base, step, limit = 10, 4, 100
    # else:
    #     base, step, limit = 10, 3, 100

    base, step, limit = 10, 3, 100

    if n == 0:
        magnitude = 0 #cannot take log(0)
    else:
        magnitude = math.log(n, base)

    order = int(round(magnitude)) // step
    return '%s%.1f %s%s' % (prefix, float(n)/base**(order*step), \
    prefixes[order], suffix)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  percent  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class percent(float):
    def __str__(self):
        return '{:.2%}'.format(self)
    def __repr__(self):
        return '{:.2%}'.format(self)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  Percent  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class Percent(float):
    def __str__(self):
        return '{:.0%}'.format(self)
    def __repr__(self):
        return '{:.0%}'.format(self)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  FLAT  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# class flat(pandas.core.frame.DataFrame):
#     def __repr__(self):
#         return ['_'.join(column) for column in self.columns.to_flat_index()]

def flat(df):
    
    df.columns = ['_'.join(column) for column in df.columns.to_flat_index()]
        
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  dtf  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def dtf():
    '''
    Displays the DateTime Formating info in Markdown
    '''
    display(Markdown('''# DateTime Formating Guide<br>
|Directive|Meaning|Example|Notes|
|:-------:|:------|:------|:----|
|%a|Weekday as locale’s abbreviated name.|Sun, Mon, …, Sat (en_US);So, Mo, …, Sa (de_DE)|(1)|
|%A|Weekday as locale’s full name.|Sunday, Monday, …, Saturday (en_US);Sonntag, Montag, …, Samstag (de_DE)|(1)|
|%w|Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.|0, 1, …, 6|
|%d|Day of the month as a zero-padded decimal number.|01, 02, …, 31|(9)|
|%b|Month as locale’s abbreviated name.|Jan, Feb, …, Dec (en_US) ;Jan, Feb, …, Dez (de_DE)|(1)|
|%B|Month as locale’s full name.|January, February, …, December (en_US); Januar, Februar, …, Dezember (de_DE)|(1)|
|%m|Month as a zero-padded decimal number.|01, 02, …, 12|(9)|
|%y|Year without century as a zero-padded decimal number.|00, 01, …, 99|(9)|
|%Y|Year with century as a decimal number.|0001, 0002, …, 2013, 2014, …, 9998, 9999|(2)|
|%H|Hour (24-hour clock) as a zero-padded decimal number.|00, 01, …, 23|(9)|
|%I|Hour (12-hour clock) as a zero-padded decimal number.|01, 02, …, 12|(9)|
|%p|Locale’s equivalent of either AM or PM.|AM, PM (en_US); am, pm (de_DE)|(1), (3)|
|%M|Minute as a zero-padded decimal number.|00, 01, …, 59|(9)|
|%S|Second as a zero-padded decimal number.|00, 01, …, 59|(4), (9)|
|%f|Microsecond as a decimal number, zero-padded to 6 digits.|000000, 000001, …, 999999|(5)|
|%z|UTC offset in the form ±HHMM[SS[.ffffff]] (empty string if the object is naive).|(empty), +0000, -0400, +1030, +063415, -030712.345216|(6)|
|%Z|Time zone name (empty string if the object is naive).|(empty), UTC, GMT|(6)|
|%j|Day of the year as a zero-padded decimal number.|001, 002, …, 366|(9)|
|%U|Week number of the year (Sunday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0.|00, 01, …, 53|(7), (9)|
|%W|Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Monday are considered to be in week 0.|00, 01, …, 53|(7), (9)|
|%c|Locale’s appropriate date and time representation.|Tue Aug 16 21:30:00 1988 (en_US); Di 16 Aug 21:30:00 1988 (de_DE)|(1)|
|%x|Locale’s appropriate date representation.|08/16/88 (None); 08/16/1988 (en_US); 16.08.1988 (de_DE)|(1)|
|%X|Locale’s appropriate time representation.|21:30:00 (en_US); 21:30:00 (de_DE)|(1)|
|%%|A literal '%' character.|%||

**Several additional directives not required by the C89 standard are included for convenience. These parameters all correspond to ISO 8601 date values.**

|Directive|Meaning|Example|Notes|
|---|---|---|---|
|%G|ISO 8601 year with century representing the year that contains the greater part of the ISO week (%V).|0001, 0002, …, 2013, 2014, …, 9998, 9999|(8)|
|%u|ISO 8601 weekday as a decimal number where 1 is Monday.|1, 2, …, 7||
|%V|ISO 8601 week as a decimal number with Monday as the first day of the week. Week 01 is the week containing Jan 4.|01, 02, …, 53|(8), (9)|
'''))
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<     >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def string_format_time_guide():
    '''
    
    '''
    
    display(Markdown('''</style>
<h1 id="strftime-format-specifiers"><code>strftime</code> Format Specifiers</h1>
<table>
<thead>
<tr class="header">
<th>Units</th>
<th>Specifier</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><strong>seconds</strong></td>
<td><code>%S</code></td>
<td>Second of the minute (00..60)</td>
</tr>
<tr class="even">
<td><strong>minutes</strong></td>
<td><code>%M</code></td>
<td>Minute of the hour (00..59)</td>
</tr>
<tr class="odd">
<td><strong>hours</strong></td>
<td><code>%H</code></td>
<td>Hour of the day, 24-hour clock (00..23)</td>
</tr>
<tr class="even">
<td></td>
<td><code>%I</code></td>
<td>Hour of the day, 12-hour clock (01..12)</td>
</tr>
<tr class="odd">
<td><strong>days</strong></td>
<td><code>%d</code></td>
<td>Day of the month</td>
</tr>
<tr class="even">
<td></td>
<td><code>%a</code></td>
<td>The abbreviated weekday name (“Sun”)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%A</code></td>
<td>The full weekday name (“Sunday”)</td>
</tr>
<tr class="even">
<td></td>
<td><code>%j</code></td>
<td>Day of the year (001..366)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%w</code></td>
<td>Day of the week, Sunday is 0 (0..6)</td>
</tr>
<tr class="even">
<td><strong>weeks</strong></td>
<td><code>%U</code></td>
<td>Week of the year, Sunday is the first day of the week (00..53)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%W</code></td>
<td>Week of the year, Monday is the first day of the week (00..53)</td>
</tr>
<tr class="even">
<td><strong>months</strong></td>
<td><code>%b</code></td>
<td>The abbreviated month name (“Jan”)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%B</code></td>
<td>The full month name (“January”)</td>
</tr>
<tr class="even">
<td></td>
<td><code>%d</code></td>
<td>Day of the month (01..31)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%m</code></td>
<td>Month of the year (01..12)</td>
</tr>
<tr class="even">
<td><strong>years</strong></td>
<td><code>%y</code></td>
<td>Year without a century (00..99)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%Y</code></td>
<td>Year with century (1999)</td>
</tr>
<tr class="even">
<td><strong>misc</strong></td>
<td><code>%z</code></td>
<td>Time zone offset (-0500)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%Z</code></td>
<td>Time zone name (“CDT”)</td>
</tr>
<tr class="even">
<td></td>
<td><code>%p</code></td>
<td>Meridian indicator (“AM” or “PM”)</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%c</code></td>
<td>The preferred local date and time representation</td>
</tr>
<tr class="even">
<td></td>
<td><code>%x</code></td>
<td>Preferred representation for the date alone, no time</td>
</tr>
<tr class="odd">
<td></td>
<td><code>%X</code></td>
<td>Preferred representation for the time alone, no date</td>
</tr>
</tbody>
</table>
</body>
</html>'''))