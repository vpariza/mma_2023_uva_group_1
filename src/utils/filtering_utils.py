###### Filtering Utility Methods ######
def apply_filters(df, filters): 
    new_df = df.copy()
    for tag, filter in filters['range'].items():
        values = {'Max': filter.Max.QueryText.text(), 'Min': filter.Min.QueryText.text() }
        for bound, input in values.items():
            try: 
                input = eval(input)
                if bound == 'Min':
                    print(bound, tag, input)
                    new_df = new_df[new_df[tag] > input]
                elif bound == 'Max':
                    print(bound, tag, input)
                    new_df = new_df[new_df[tag] < input]
            except SyntaxError:
                print(bound, tag, 'not provided')
            except NameError:
                print('invalid input for ', bound, tag)
    for tag, filter in filters['combo'].items():
        if filter.Filter.currentText() != '':
            print(tag)
            if (tag == 'bedrooms') or (tag == 'number_of_rooms'):
                print('filter edited to min')
                new_df = new_df[new_df[tag] >= eval(filter.Filter.currentText())] 
            else:
                new_df = new_df[new_df[tag] == filter.Filter.currentText()]
    return new_df