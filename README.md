# RAG-Based-Knowledge-Assistant

User Query
   ↓
FAISS (semantic search)
   ↓
Relevant chunks
   ↓
TinyLlama (LLM)
   ↓
Final answer

# checkout the results
query = "What items are in invoice 0012820?"

--- Retrieved 0 ---
INVOICE 0012820
Date: 27.08.2019

ConIncorporated
305 Fleet Rd
Fleet, Hampshire County,
GU51 3BU
012 5261 2116
admin@conincorp.co.uk
 SHIP TO : Caitlin Roberts
Awthentikz
89 Annfield Rd
BEARLEY, CV37 7GQ
079 0608 3650
Customer ID: CN0044
BILL TO:  Caitlin Roberts
Awthentikz
89 Annfi

--- Retrieved 1 ---
16.12.2021

Invoice No. 1213


2
Queen Anne's Lace
Qty. 32          6.05

193.60
Alaskan Douglasia
Qty. 122          4.36

531.92
Deer Sedge
Qty. 23          3.82

87.86
Indian Tobacco
Qty. 2          7.25


--- Retrieved 2 ---
Zencorporations
Tel + 49 228 698629
Fax + 49 228 698629

Thomas-Mann-Strasse 38
Bonn, 53111

zencorporations.de
contact@zencorporations.de



16.12.2021

 Invoice No. 1213
To
La Galerie
4 Rue Courtois
Lille, Nord, 59000
Ship To
Same as recipient
Instructions
None

=== FINAL ANSWER ===

Invoice 0012820 contains items such as 10-700 - Exterior Protection (10), 1-515 - Temporary Lighting (29), 11-060 - Theater and Stage Equipment (17), 1-600 - Product Requirements (Scope of Work) (20), and 16.12.2021.