import os
import random
import pandas as pd
import json

# Define output directory
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Document classes
CLASSES = ["invoice", "receipt", "resume", "letter", "scientific_report", "legal_contract"]

# OCR Noise Generator
def add_ocr_noise(text, noise_level=0.1):
    chars = list(text)
    n_changes = int(len(chars) * noise_level)
    noisy_chars = ['|', ']', '[', '_', '-', '~', '0', '1', 'l', 'I']
    
    for _ in range(n_changes):
        idx = random.randint(0, len(chars) - 1)
        action = random.choice(['replace', 'swap', 'drop', 'insert'])
        if action == 'replace' and chars[idx].isalnum():
            chars[idx] = random.choice(noisy_chars)
        elif action == 'swap' and idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif action == 'drop' and len(chars) > 10:
            chars[idx] = ''
        elif action == 'insert':
            chars[idx] = chars[idx] + random.choice(noisy_chars)
            
    # Simulate line break & spacing OCR artifacts
    res = "".join(chars)
    if random.random() < 0.3:
        res = res.replace(" ", "  ")
    if random.random() < 0.3:
        res = res.replace("\n", " \n ")
    return res

# Templates generator per class
companies = ["Acme Corp", "Apex Solutions", "Global Tech Inc", "Nexus Logistics", "Vanguard Industries", "Starlight Media", "Quantum Systems", "Horizon Health"]
names = ["John Smith", "Alice Johnson", "Robert Miller", "Emma Watson", "David Lee", "Sophia Martinez", "Michael Brown", "Sarah Wilson"]
cities = ["New York, NY", "San Francisco, CA", "Chicago, IL", "Austin, TX", "Seattle, WA", "Boston, MA", "London, UK", "Toronto, ON"]
skills_list = ["Python, PyTorch, Machine Learning, SQL, Docker, AWS, Kubernetes", 
               "Project Management, Agile, Scrum, Budgeting, Leadership, JIRA", 
               "Java, Spring Boot, Microservices, REST APIs, PostgreSQL, Git",
               "Data Analysis, Tableau, R, Statistics, Pandas, PowerBI",
               "Graphic Design, Photoshop, Illustrator, UI/UX, Figma, HTML/CSS"]
universities = ["MIT", "Stanford University", "UC Berkeley", "Harvard University", "Carnegie Mellon", "Oxford University"]

def generate_invoice():
    company = random.choice(companies)
    inv_num = f"INV-2026-{random.randint(1000, 9999)}"
    amount = f"${random.randint(100, 15000)}.{random.randint(10, 99):02d}"
    due_date = f"2026-09-{random.randint(1, 28):02d}"
    text = f"""INVOICE
{company}
Invoice Number: {inv_num}
Date: 2026-08-{random.randint(1, 10):02d}
Due Date: {due_date}
Bill To: {random.choice(companies)}
Items:
1. Professional Services / Consulting - {amount}
2. Software Subscriptions & Server Maintenance - $450.00
3. Tax (8.5%): ${random.randint(20, 200)}.00
TOTAL AMOUNT DUE: {amount}
Payment Methods: Wire Transfer, Credit Card. Please remit payment by due date."""
    return text

def generate_receipt():
    vendor = random.choice(["Target Retail", "Walmart Supercenter", "Starbucks Coffee", "Best Buy Electronics", "Trader Joe's", "Home Depot", "Shell Gas Station"])
    items = ["Milk 1Gal $4.29", "Coffee Beans $14.99", "USB-C Cable $19.99", "Organic Apples $5.50", "Paper Towels $11.20", "Croissant $3.75", "Gasoline Regular $45.00"]
    item_sample = "\n".join(random.sample(items, k=random.randint(2, 4)))
    total = f"${random.randint(10, 150)}.{random.randint(10, 99):02d}"
    text = f"""*** RECEIPT ***
{vendor}
Store #{random.randint(100, 999)} - {random.choice(cities)}
Date: 2026-08-{random.randint(1, 8):02d} Time: {random.randint(8, 21)}:{random.randint(10, 59)}
--------------------------------
{item_sample}
SUBTOTAL: {total}
TAX (8%): ${random.randint(1, 10)}.50
TOTAL: {total}
AUTH CODE: {random.randint(100000, 999999)} VISA ENDING IN *{random.randint(1000, 9999)}
THANK YOU FOR SHOPPING WITH US!"""
    return text

def generate_resume():
    candidate = random.choice(names)
    title = random.choice(["Senior Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", "UX Designer", "Financial Analyst"])
    uni = random.choice(universities)
    skills = random.choice(skills_list)
    text = f"""{candidate}
Email: {candidate.lower().replace(' ', '.')}@email.com | Phone: (555) {random.randint(100,999)}-{random.randint(1000,9999)}
Location: {random.choice(cities)}

OBJECTIVE
Experienced {title} with over {random.randint(3, 12)} years of proven track record in technology and management.

WORK EXPERIENCE
{random.choice(companies)} - {title} (2022 - Present)
- Led a team of {random.randint(4, 15)} engineers building scalable cloud systems.
- Reduced system latency by {random.randint(20, 50)}% and optimized deployment workflows.

Education: B.S. in Computer Science - {uni}
Technical Skills: {skills}
Certifications: AWS Certified Solutions Architect, Scrum Master"""
    return text

def generate_letter():
    sender = random.choice(names)
    company = random.choice(companies)
    recipient = random.choice(names)
    text = f"""{company}
100 Business Parkway, Suite {random.randint(100, 900)}
{random.choice(cities)}

Date: August {random.randint(1, 10)}, 2026

Dear {recipient},

I am writing to formally communicate our quarterly performance updates regarding our ongoing joint project. We have evaluated the operational metrics and are pleased to report significant progress across all deliverables.

Should you have any questions or require additional documentation, please do not hesitate to contact my office. We look forward to our continued partnership and successful execution in the upcoming fiscal quarter.

Sincerely,

{sender}
Executive Vice President
{company}"""
    return text

def generate_scientific_report():
    title = random.choice([
        "Evaluation of Transformer Architectures in OCR Text Classification",
        "Deep Learning Applications in Automated Medical Imaging Analysis",
        "A Novel Approach to Distributed Ledger Consensus Mechanisms",
        "Experimental Analysis of Quantum Annealing Algorithms for Graph Optimization",
        "Comparative Benchmark of Lightweight Neural Networks on Edge Devices"
    ])
    authors = f"{random.choice(names)}, {random.choice(names)}, and {random.choice(names)}"
    text = f"""RESEARCH REPORT / JOURNAL OF APPLIED AI
Title: {title}
Authors: {authors}
Affiliation: Department of Computer Science, {random.choice(universities)}

ABSTRACT
In this study, we present a novel framework for evaluating deep learning model efficiency under real-world noise conditions. We conduct extensive experiments across diverse datasets, analyzing convergence rates, precision, recall, and F1-score performance. 

1. INTRODUCTION
Recent advances in neural architecture design have driven significant gains in natural language processing tasks. However, computational resource limits necessitate lightweight models for real-time inference.

2. METHODOLOGY AND EXPERIMENTAL SETUP
We utilize a dataset of {random.randint(5000, 50000)} samples processed via custom tokenization. Models were trained using AdamW optimizer with a learning rate of 2e-5 over 10 epochs.

3. RESULTS AND DISCUSSION
Our empirical evaluation demonstrates a {random.randint(90, 98)}% accuracy while reducing inference latency by 45% compared to baseline models."""
    return text

def generate_legal_contract():
    party_a = random.choice(companies)
    party_b = random.choice(companies)
    doc_id = f"AGR-2026-{random.randint(100, 999)}"
    text = f"""NON-DISCLOSURE AND SERVICE AGREEMENT
Contract Ref: {doc_id}

This Agreement is entered into on this {random.randint(1, 28)}th day of August 2026, by and between {party_a} ("Disclosing Party") and {party_b} ("Receiving Party").

1. CONFIDENTIAL INFORMATION
The Receiving Party agrees to hold in confidence all proprietary technical, financial, and business information disclosed by Party A. Confidential Information shall not be disclosed to any third party without prior written authorization.

2. OBLIGATIONS AND GOVERNING LAW
This agreement shall be governed by and construed in accordance with the laws of the State of California. Any disputes arising hereunder shall be resolved via binding arbitration.

IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first above written.

Signed: 
_________________________ ({party_a})
_________________________ ({party_b})"""
    return text

GENERATORS = {
    "invoice": generate_invoice,
    "receipt": generate_receipt,
    "resume": generate_resume,
    "letter": generate_letter,
    "scientific_report": generate_scientific_report,
    "legal_contract": generate_legal_contract
}

def build_dataset(samples_per_class=200):
    data = []
    print(f"Generating dataset with {samples_per_class} samples per class across {len(CLASSES)} classes...")
    
    for cls in CLASSES:
        gen_fn = GENERATORS[cls]
        for _ in range(samples_per_class):
            clean_text = gen_fn()
            # Apply OCR noise to 70% of samples
            if random.random() < 0.7:
                text = add_ocr_noise(clean_text, noise_level=random.uniform(0.02, 0.08))
            else:
                text = clean_text
            data.append({"text": text, "label": cls})
            
    random.shuffle(data)
    df = pd.DataFrame(data)
    
    # Train / Val / Test Split: 70% Train, 15% Val, 15% Test
    total = len(df)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATASET_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)
    
    # Save label mappings
    label2id = {label: idx for idx, label in enumerate(CLASSES)}
    id2label = {idx: label for idx, label in enumerate(CLASSES)}
    with open(os.path.join(DATASET_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        
    print(f"Dataset generated successfully!")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples:   {len(val_df)}")
    print(f"  Test samples:  {len(test_df)}")

if __name__ == "__main__":
    random.seed(42)
    build_dataset(samples_per_class=150) # Total 900 samples (630 train, 135 val, 135 test)
