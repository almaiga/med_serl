injector_realism = 1.0
assessor_accuracy = 1.0

gamma = 0.99
final_assessor = assessor_accuracy
final_injector = injector_realism - final_assessor

return_assessor = final_assessor
return_injector = final_injector + gamma * final_assessor

print(f"Assessor Return: {return_assessor}")
print(f"Injector Return: {return_injector}")
