from transformers import AutoModel


provence = AutoModel.from_pretrained("naver/provence-reranker-debertav3-v1", trust_remote_code=True)
print(provence)

context = "Shepherd’s pie. History. In early cookery books, the dish was a means of using leftover roasted meat of any kind, and the pie dish was lined on the sides and bottom with mashed potato, as well as having a mashed potato crust on top. Variations and similar dishes. Other potato-topped pies include: The modern ”Cumberland pie” is a version with either beef or lamb and a layer of bread- crumbs and cheese on top. In medieval times, and modern-day Cumbria, the pastry crust had a filling of meat with fruits and spices.. In Quebec, a varia- tion on the cottage pie is called ”Paˆte ́ chinois”. It is made with ground beef on the bottom layer, canned corn in the middle, and mashed potato on top.. The ”shepherdess pie” is a vegetarian version made without meat, or a vegan version made without meat and dairy.. In the Netherlands, a very similar dish called ”philosopher’s stew” () often adds ingredients like beans, apples, prunes, or apple sauce.. In Brazil, a dish called in refers to the fact that a manioc puree hides a layer of sun-dried meat."
question = 'What goes on the bottom of Shepherd’s pie?'

provence_output = provence.process(question, context)
print(f"Provence Output: {provence_output}")
# Provence Output: {'reranking_score': 3.022725, pruned_context': 'In early cookery books, the dish was a means of using leftover roasted meat of any kind, and the pie dish was lined on the sides and bottom with mashed potato, as well as having a mashed potato crust on top.']]

