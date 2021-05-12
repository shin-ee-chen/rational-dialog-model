from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer

my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

print(my_tokenizer.encode('[RMASK]').tokens)
print(my_tokenizer.encode('[RMASK]').ids)