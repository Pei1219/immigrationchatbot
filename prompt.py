prompt = f"""
You will be provided with a specific immigration rule summary, the full text of the rule, and relevant context. 
Your task is to generate a question-answer pair based on the rule and its application in real-world scenarios.

The question should be formulated to reflect a typical query that someone seeking immigration advice might have, and the answer must be accurate, aligned with the provided rule, and formatted clearly.

Ensure that the answer either directly quotes the relevant part of the rule or provides a precise interpretation that remains legally sound. Both the question and the answer should be structured to help the user easily understand the application of the rule.

Example 1:
Rule Overview: "This rule pertains to the requirements for a student visa holder to switch to a skilled worker visa after graduation."
Full Rule Text: "A student visa holder may apply for a skilled worker visa if they have completed their degree at an approved UK institution, have a job offer from a licensed sponsor, and meet the minimum salary threshold of £25,600 or the 'going rate' for the job, whichever is higher."
Question: "Can a student visa holder in the UK apply for a skilled worker visa after completing their degree?"
Answer: "Yes, a student visa holder can apply for a skilled worker visa after graduation, provided they have a job offer from a licensed sponsor and meet the minimum salary requirement of £25,600 or the 'going rate' for the job."

Example 2:
Rule Overview: "This rule details the financial requirements for family members of a UK citizen applying for a visa."
Full Rule Text: "The applicant must demonstrate that they have access to sufficient funds to support themselves and any dependents without recourse to public funds. The minimum income requirement is £18,600 per year for the applicant alone, with additional funds required for each dependent."
Question: "What is the minimum income requirement for a UK citizen's spouse applying for a visa?"
Answer: "The minimum income requirement for a UK citizen's spouse applying for a visa is £18,600 per year. Additional income is required if there are dependents."

Now, based on the provided rule, generate a similar question-answer pair.

Rule Overview: "{rule_overview}"
Full Rule Text: "{full_rule_text}"
Question: [Generate a relevant question based on the rule]
Answer: [Provide a clear, accurate answer based on the rule]

Make sure the question directly relates to the rule, and the answer is either a direct quote or a precise, legally sound interpretation. Return the result in the format: 'Question: [Your question] Answer: [Your answer]'
"""