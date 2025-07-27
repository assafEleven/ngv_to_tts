# ================================================================
# 📦 Cell 10: Query Table Router
# Purpose: Route user queries to the best-fit table(s) using GLOBAL_SCHEMA_INDEX
# Depends on: BigQuery auth (Cell 02), Schema Index (Cell 09)
# ================================================================

from typing import List, Dict, Tuple, Optional
import pandas as pd
import difflib

# This function expects GLOBAL_SCHEMA_INDEX to be available (from Cell 09)
def route_query_to_tables(required_fields: List[str], alias_map: Dict[str, List[str]], min_match: float = 0.5, verbose: bool = True) -> Tuple[List[Dict], Optional[Dict], float]:
    """
    Route a query to all candidate tables in GLOBAL_SCHEMA_INDEX based on required fields and aliases.
    Returns (ranked_candidates, best_mapping, best_score)
    Each candidate is a dict with: table info, mapping, score, missing, reasons
    """
    global GLOBAL_SCHEMA_INDEX
    if GLOBAL_SCHEMA_INDEX.empty:
        if verbose:
            print("⚠️  GLOBAL_SCHEMA_INDEX is empty - build it in Cell 09 first.")
        return [], None, 0
    table_columns = GLOBAL_SCHEMA_INDEX.groupby(['project', 'dataset', 'table'])['column'].apply(list).reset_index()
    candidates = []
    for _, row in table_columns.iterrows():
        mapping = {}
        match_count = 0
        reasons = []
        missing = []
        score = 0
        for field in required_fields:
            found = None
            found_type = None
            # Direct match
            for col in row['column']:
                if col.lower() == field.lower():
                    found = col
                    found_type = 'direct'
                    break
            # Alias match
            if not found:
                for alias in alias_map.get(field, []):
                    for col in row['column']:
                        if alias.lower() == col.lower():
                            found = col
                            found_type = 'alias'
                            break
                    if found:
                        break
            # Fuzzy/partial match
            if not found:
                for alias in alias_map.get(field, []):
                    for col in row['column']:
                        if alias.lower() in col.lower() or col.lower() in alias.lower():
                            found = col
                            found_type = 'fuzzy'
                            break
                    if found:
                        break
            # Name similarity (difflib)
            if not found:
                close = difflib.get_close_matches(field.lower(), [c.lower() for c in row['column']], n=1, cutoff=0.7)
                if close:
                    idx = [c.lower() for c in row['column']].index(close[0])
                    found = row['column'][idx]
                    found_type = 'similarity'
            # Data type compatibility (if available)
            # (Not implemented: would require joining with data_type info)
            mapping[field] = found
            if found:
                match_count += 1
                reasons.append(f"{field} → `{found}` ({found_type})")
            else:
                missing.append(field)
                reasons.append(f"{field} → ❌ (not found)")
        # Confidence score: weighted by # matched, match type, and penalty for missing
        base_score = match_count / len(required_fields)
        # Bonus for direct/alias, penalty for fuzzy/similarity
        bonus = 0
        for field in required_fields:
            if mapping[field]:
                if mapping[field].lower() == field.lower():
                    bonus += 0.15
                elif any(mapping[field].lower() == a.lower() for a in alias_map.get(field, [])):
                    bonus += 0.10
                else:
                    bonus += 0.05
        score = min(1.0, base_score + bonus - 0.05*len(missing))
        candidates.append({
            'project': row['project'],
            'dataset': row['dataset'],
            'table': row['table'],
            'full_table_id': f"{row['project']}.{row['dataset']}.{row['table']}",
            'mapping': mapping,
            'score': score,
            'missing': missing,
            'reasons': reasons
        })
    # Rank candidates
    ranked = sorted([c for c in candidates if c['score'] > 0], key=lambda x: x['score'], reverse=True)
    if not ranked or ranked[0]['score'] < min_match:
        if verbose:
            print(f"❌ No table found with confidence >= {int(min_match*100)}% for fields: {required_fields}")
            if ranked:
                print(f"   Top candidate: {ranked[0]['full_table_id']} (score: {int(ranked[0]['score']*100)}%)")
                print(f"   Missing: {', '.join(ranked[0]['missing'])}")
            else:
                print("   No candidates matched any required fields.")
            print("   You may provide a table or column hint.")
        return ranked, None, 0
    # Conversational transparency
    best = ranked[0]
    if verbose:
        print(f"✅ Using `{best['full_table_id']}` (confidence: {int(best['score']*100)}%, rank: 1)")
        for reason in best['reasons']:
            print(f"   {reason}")
        if best['missing']:
            print(f"   Missing: {', '.join(best['missing'])}")
        if len(ranked) > 1:
            print(f"   {len(ranked)-1} additional candidate(s) found.")
    return ranked, best['mapping'], best['score']

# --- Alias: route_query_to_table for compatibility ---
def route_query_to_table(required_fields: List[str], alias_map: Dict[str, List[str]], min_match: float = 0.5, verbose: bool = True):
    """
    Wrapper for route_query_to_tables that returns only the best candidate (row, mapping, score).
    """
    ranked, mapping, score = route_query_to_tables(required_fields, alias_map, min_match, verbose)
    if not ranked or not mapping:
        return None, None, 0
    best = ranked[0]
    return best, mapping, score 