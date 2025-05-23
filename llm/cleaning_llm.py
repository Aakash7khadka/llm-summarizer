import pandas as pd
import json
from tqdm import tqdm
from ollama_client import generate_llm_summary
from generate_summaries import clean_llm_output
import matplotlib.pyplot as plt

llm_json = 'data/summaries_20news_llm_2.json'

llm_write_json = 'data/summaries_20news_llm_2.json'

# df_idx = "15373 13842 10072 8406 6714 6361 2080 437 62"
# df_idx_empty = ['184', '508', '816', '4749', '4860', '7341', '10024', '10337', '12704', '13204', '14224', '15046', '15106', '15656', '15969']
# df_idx = df_idx.split(" ")
# for id in range(len(df_idx)):
#     df_idx[id] = int(df_idx[id])

# df = pd.read_csv("data/cleaned_20news_light.csv")

with open(llm_json, 'r', encoding="utf-8") as file:
    llm_summaries = json.load(file)


# Create histogram
# llm_summaries_list = list(llm_summaries.values())
# llm_summaries_lens = [len(summary.split()) for summary in llm_summaries_list]
# counts, bin_edges, _ = plt.hist(llm_summaries_lens, bins=100, edgecolor='black')

# # Annotate each bin with its range
# for i in range(len(counts)):
#     bin_range = f"{(bin_edges[i] + bin_edges[i+1])/2:.1f}"
#     plt.text((bin_edges[i] + bin_edges[i+1]) / 2, counts[i], bin_range,
#              ha='center', va='bottom', fontsize=9)

# # Labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram with Bin Ranges')

# plt.show()




# for doc_id, text in tqdm(llm_summaries.items(), total=len(llm_summaries)):
#     if len(text.split()) > 100:
#         try:
#             llm_raw = generate_llm_summary(text, doc_id=doc_id)
#             llm_summary = clean_llm_output(llm_raw)
#             if llm_summary:
#                 llm_summaries[doc_id] = llm_summary
#         except Exception as e:
#             print(f"[LLM failed] ID={doc_id} | {e}")




# for doc_id in df_idx:
#     text = str(df.iloc[doc_id].get('text', "")).strip()
#     print(text)
#     print('-'*100)
#     print('-'*100)
#     print('-'*100)
    # try:
    #     llm_raw = generate_llm_summary(text, doc_id=doc_id)
    #     llm_summary = clean_llm_output(llm_raw)
    #     if llm_summary:
    #         llm_summaries[doc_id] = llm_summary
    #         print(f"[LLM successfukly generated] ID={doc_id}")
    # except Exception as e:
    #     print(f"[LLM failed] ID={doc_id} | {e}")

deleted_ids = []
iter_summary_dict = llm_summaries.copy().items()
for doc_id, text in tqdm(iter_summary_dict, total=len(iter_summary_dict)):
    lower_text = text.lower()
    if lower_text.startswith("please provide the text"):
        deleted_ids.append(doc_id)
        del llm_summaries[doc_id]

print(f"Deleted {len(deleted_ids)} summaries.")
print(deleted_ids)

with open(llm_write_json, 'w', encoding='utf-8') as f:
    json.dump(llm_summaries, f, ensure_ascii=False, indent=2)