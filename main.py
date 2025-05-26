# GitHub Data Collector + Analysis for Code Review Impact Study
# Analyzes: Bug Density, Review Depth (comments per PR), and Time to Bug Resolution

from github import Github
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import time
import numpy as np
import shutil

# ----------------------------
# STEP 0: Load environment
# ----------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
g = Github(GITHUB_TOKEN)

# ----------------------------
# STEP 1: Define repositories (3 with code review, 3 without)
# ----------------------------
REPOS = [
    # Projects WITH mandatory code reviews (use PRs extensively)
    ("tensorflow/tensorflow", True),
    ("microsoft/vscode", True), 
    ("facebook/react", True),
    
    # Projects WITHOUT mandatory code reviews (allow direct commits)
    ("torvalds/linux", False),
    ("git/git", False),
    ("vim/vim", False)
]

# ----------------------------
# STEP 2: Enhanced data collection function
# ----------------------------
def collect_repo_data(repo_full_name, uses_code_review):
    """Collect comprehensive data from repository"""
    repo = g.get_repo(repo_full_name)
    print(f"\n🔍 Collecting data from {repo_full_name}...")
    
    # Initialize data structure
    data = {
        "repo": repo_full_name,
        "uses_code_review": uses_code_review,
        "bug_issues": 0,
        "resolved_bugs": 0,
        "avg_bug_resolution_days": 0,
        "median_bug_resolution_days": 0,
        "merged_prs": 0,
        "total_reviews": 0,
        "total_review_comments": 0,
        "avg_review_depth": 0,  # comments per PR
        "commits": 0,
        "direct_commits": 0,
        "lines_of_code": 0,
        "bug_density": 0,
        "avg_reviews_per_pr": 0
    }
    
    # Track bug resolution times
    bug_resolution_times = []
    
    try:
        # 1. Enhanced bug analysis with resolution times
        print("  📊 Analyzing bugs and resolution times...")
        bug_labels = ["bug", "Bug", "BUG", "defect", "issue"]
        total_bugs = 0
        resolved_bugs = 0
        
        # First try labeled bugs
        for label_name in bug_labels:
            try:
                issues = repo.get_issues(state='all', labels=[label_name])
                issue_count = 0
                
                for issue in issues:
                    if issue_count >= 25:  # Cap at 25 per label
                        break
                    
                    total_bugs += 1
                    issue_count += 1
                    
                    # Calculate resolution time if bug is closed
                    if issue.state == 'closed' and issue.closed_at and issue.created_at:
                        resolution_time = (issue.closed_at - issue.created_at).days
                        bug_resolution_times.append(resolution_time)
                        resolved_bugs += 1
                        
                if issue_count > 0:
                    print(f"    Found {issue_count} issues with label '{label_name}'")
                    
            except Exception as e:
                print(f"    ⚠️ Error with label '{label_name}': {e}")
                continue
        
        # If no labeled bugs found, search by keywords
        if total_bugs == 0:
            print("  🔍 No labeled bugs found, searching by keywords...")
            try:
                bug_issues = repo.get_issues(state='all')
                bug_keywords = ['bug', 'error', 'crash', 'fail', 'broken', 'fix']
                
                count = 0
                for issue in bug_issues:
                    if count >= 50:  # Check more issues for keyword search
                        break
                    
                    count += 1
                    
                    # Check if title contains bug keywords
                    if any(keyword in issue.title.lower() for keyword in bug_keywords):
                        total_bugs += 1
                        
                        # Calculate resolution time if closed
                        if issue.state == 'closed' and issue.closed_at and issue.created_at:
                            resolution_time = (issue.closed_at - issue.created_at).days
                            bug_resolution_times.append(resolution_time)
                            resolved_bugs += 1
                    
                    if total_bugs >= 25:  # Stop after finding 25 bugs
                        break
                        
            except Exception as e:
                print(f"    ⚠️ Error searching by keywords: {e}")
        
        data["bug_issues"] = total_bugs
        data["resolved_bugs"] = resolved_bugs
        
        # Calculate bug resolution metrics
        if bug_resolution_times:
            data["avg_bug_resolution_days"] = round(np.mean(bug_resolution_times), 1)
            data["median_bug_resolution_days"] = round(np.median(bug_resolution_times), 1)
            print(f"  ✅ Bugs: {total_bugs} total, {resolved_bugs} resolved")
            print(f"  ⏱️ Avg resolution: {data['avg_bug_resolution_days']} days, Median: {data['median_bug_resolution_days']} days")
        else:
            print(f"  ✅ Total bugs found: {total_bugs} (no resolution times available)")
        
    except Exception as e:
        print(f"  ❌ Error analyzing bugs: {e}")
    
    try:
        # 2.Pull Request analysis with review depth
        print("  📊 Analyzing pull requests and review depth...")
        pulls = repo.get_pulls(state='closed', sort='updated', direction='desc')
        
        merged_prs = []
        total_reviews = 0
        total_comments = 0
        pr_commit_shas = set()
        
        pr_count = 0
        merged_count = 0
        
        for pr in pulls:
            if merged_count >= 25:  # Stop after 25 merged PRs
                break
                
            pr_count += 1
                
            if pr.merged:
                merged_prs.append(pr)
                merged_count += 1
                
                try:
                    # Count reviews
                    reviews = pr.get_reviews()
                    pr_review_count = reviews.totalCount
                    total_reviews += pr_review_count
                    
                    # Count review comments (this is our review depth metric)
                    review_comments = pr.get_review_comments()
                    pr_comment_count = review_comments.totalCount
                    total_comments += pr_comment_count
                    
                    # Also count issue comments on the PR
                    issue_comments = pr.get_issue_comments()
                    pr_issue_comments = issue_comments.totalCount
                    total_comments += pr_issue_comments
                    
                    # Track PR commits
                    commit_count = 0
                    for commit in pr.get_commits():
                        pr_commit_shas.add(commit.sha)
                        commit_count += 1
                        if commit_count >= 10:  # Cap commits per PR
                            break
                            
                except Exception as e:
                    print(f"    ⚠️ Error processing PR #{pr.number}: {e}")
                    continue
                
                # Progress update
                if merged_count % 10 == 0:
                    print(f"    Processed {merged_count}/25 merged PRs...")
            
            # Safety break
            if pr_count >= 100:
                print(f"    Checked {pr_count} PRs, found {merged_count} merged")
                break
        
        data["merged_prs"] = len(merged_prs)
        data["total_reviews"] = total_reviews
        data["total_review_comments"] = total_comments
        
        # Calculate review depth (comments per PR)
        if len(merged_prs) > 0:
            data["avg_review_depth"] = round(total_comments / len(merged_prs), 2)
            data["avg_reviews_per_pr"] = round(total_reviews / len(merged_prs), 2)
        
        print(f"  ✅ Merged PRs: {len(merged_prs)}")
        print(f"  💬 Review depth: {data['avg_review_depth']} comments per PR")
        print(f"  👥 Reviews: {data['avg_reviews_per_pr']} reviews per PR")
        
    except Exception as e:
        print(f"  ❌ Error analyzing PRs: {e}")
    
    try:
        # 3. Analyze commits (unchanged from original)
        print("  📊 Analyzing commits...")
        default_branch = repo.default_branch
        commits = repo.get_commits(sha=default_branch)
        
        commit_list = []
        commit_count = 0
        for commit in commits:
            if commit_count >= 25:
                break
            commit_list.append(commit)
            commit_count += 1
        
        total_commits = len(commit_list)
        direct_commits = [c for c in commit_list if c.sha not in pr_commit_shas]
        direct_commit_count = len(direct_commits)
        
        data["commits"] = total_commits
        data["direct_commits"] = direct_commit_count
        print(f"  ✅ Total commits: {total_commits}, Direct commits: {direct_commit_count}")
        
    except Exception as e:
        print(f"  ❌ Error analyzing commits: {e}")
    
    try:
        # 4. Get lines of code (unchanged from original)
        print("  📊 Counting lines of code...")
        languages = repo.get_languages()
        total_loc = sum(languages.values())
        data["lines_of_code"] = total_loc
        print(f"  ✅ Total LOC: {total_loc:,}")
        
    except Exception as e:
        print(f"  ❌ Error counting LOC: {e}")
    
    # 5. Calculate bug density
    if data["lines_of_code"] > 0:
        data["bug_density"] = round(data["bug_issues"] / (data["lines_of_code"] / 1000), 3)
        print(f"  ✅ Bug density: {data['bug_density']} bugs per KLOC")
    
    return data

# ----------------------------
# STEP 3: Collect data from all repositories
# ----------------------------
print("🚀 Starting enhanced data collection...")
all_data = []

for repo_name, uses_review in REPOS:
    try:
        data = collect_repo_data(repo_name, uses_review)
        all_data.append(data)
        print(f"✅ Successfully collected data from {repo_name}")
        
        # Rate limit protection
        time.sleep(3)
        
    except Exception as e:
        print(f"❌ Failed to collect from {repo_name}: {e}")
        continue

# ----------------------------
# STEP 4: Create DataFrame and save results
# ----------------------------
folder_path = 'results'

if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    print(f"Deleted existing folder: {folder_path}")

os.makedirs(folder_path)
print(f"Created new folder: {folder_path}")

if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv("results/analysis.csv", index=False)
    
    print("\n" + "="*80)
    print("📊 ENHANCED DATA COLLECTION SUMMARY")
    print("="*80)
    
    # Display key metrics in a formatted way
    print(f"{'Repository':<20} {'Code Review':<12} {'Bug Density':<12} {'Review Depth':<13} {'Resolution Days':<15}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        repo_name = row['repo'].split('/')[-1]
        uses_review = "Yes" if row['uses_code_review'] else "No"
        bug_density = f"{row['bug_density']:.3f}"
        review_depth = f"{row['avg_review_depth']:.2f}"
        resolution_days = f"{row['avg_bug_resolution_days']:.1f}" if row['avg_bug_resolution_days'] > 0 else "N/A"
        
        print(f"{repo_name:<20} {uses_review:<12} {bug_density:<12} {review_depth:<13} {resolution_days:<15}")
    
    # ----------------------------
    # STEP 5: Enhanced Statistical Analysis
    # ----------------------------
    print("\n" + "="*80)
    print("📈 COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*80)
    
    # Split data by code review usage
    with_review = df[df['uses_code_review'] == True]
    without_review = df[df['uses_code_review'] == False]
    
    print(f"\nProjects WITH code review: {len(with_review)}")
    print(f"Projects WITHOUT code review: {len(without_review)}")
    
    def analyze_metric(metric_name, column_name, unit=""):
        """Helper function to analyze a specific metric"""
        print(f"\n🔍 {metric_name} Analysis:")
        
        with_data = with_review[column_name].dropna()
        without_data = without_review[column_name].dropna()
        
        if len(with_data) > 0 and len(without_data) > 0:
            print(f"  With Code Review: Mean={with_data.mean():.3f}{unit}, Median={with_data.median():.3f}{unit}")
            print(f"  Without Code Review: Mean={without_data.mean():.3f}{unit}, Median={without_data.median():.3f}{unit}")
            
            # Statistical tests
            try:
                t_stat, p_val = ttest_ind(with_data, without_data)
                print(f"  T-Test: t={t_stat:.3f}, p={p_val:.4f}")
                
                u_stat, u_p = mannwhitneyu(with_data, without_data, alternative='two-sided')
                print(f"  Mann-Whitney U: U={u_stat:.3f}, p={u_p:.4f}")
                
                if p_val < 0.05 or u_p < 0.05:
                    print("  ✅ Statistically significant difference found!")
                else:
                    print("  ⚠️ No statistically significant difference")
                    
            except Exception as e:
                print(f"  ❌ Statistical tests failed: {e}")
        else:
            print("  ⚠️ Insufficient data for comparison")
    
    # Analyze all three main metrics
    analyze_metric("Bug Density", "bug_density", " bugs/KLOC")
    analyze_metric("Review Depth", "avg_review_depth", " comments/PR")
    analyze_metric("Bug Resolution Time", "avg_bug_resolution_days", " days")
    
    # ----------------------------
    # STEP 6: Enhanced Visualizations
    # ----------------------------
    print("\n📊 Creating enhanced visualizations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Code Review Impact Analysis: Three Key Metrics', fontsize=16, fontweight='bold')
    
    # 1. Bug Density Comparison
    sns.boxplot(data=df, x='uses_code_review', y='bug_density', ax=axes[0,0])
    sns.stripplot(data=df, x='uses_code_review', y='bug_density', 
    color='red', alpha=0.7, size=8, ax=axes[0,0])
    axes[0,0].set_title('Bug Density: With vs Without Code Review')
    axes[0,0].set_xlabel('Uses Mandatory Code Review')
    axes[0,0].set_ylabel('Bug Density (bugs per 1K LOC)')
    axes[0,0].set_xticklabels(['No', 'Yes'])
    
    # 2. Review Depth Comparison
    sns.boxplot(data=df, x='uses_code_review', y='avg_review_depth', ax=axes[0,1])
    sns.stripplot(data=df, x='uses_code_review', y='avg_review_depth', 
    color='blue', alpha=0.7, size=8, ax=axes[0,1])
    axes[0,1].set_title('Review Depth: Comments per Pull Request')
    axes[0,1].set_xlabel('Uses Mandatory Code Review')
    axes[0,1].set_ylabel('Average Comments per PR')
    axes[0,1].set_xticklabels(['No', 'Yes'])
    
    # 3. Bug Resolution Time Comparison
    df_with_resolution = df[df['avg_bug_resolution_days'] > 0]
    if len(df_with_resolution) > 0:
        sns.boxplot(data=df_with_resolution, x='uses_code_review', y='avg_bug_resolution_days', ax=axes[1,0])
        sns.stripplot(data=df_with_resolution, x='uses_code_review', y='avg_bug_resolution_days', 
        color='green', alpha=0.7, size=8, ax=axes[1,0])
        axes[1,0].set_title('Bug Resolution Time')
        axes[1,0].set_xlabel('Uses Mandatory Code Review')
        axes[1,0].set_ylabel('Average Resolution Time (days)')
        axes[1,0].set_xticklabels(['No', 'Yes'])
    else:
        axes[1,0].text(0.5, 0.5, 'Insufficient Resolution\nTime Data', 
        ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
        axes[1,0].set_title('Bug Resolution Time - No Data Available')
    
    # 4. Summary metrics by repository
    repo_names = [repo.split('/')[-1] for repo in df['repo']]
    colors = ['green' if x else 'red' for x in df['uses_code_review']]
    
    # Create a composite score (lower is better)
    df['composite_score'] = df['bug_density'] + (df['avg_bug_resolution_days'] / 10) - (df['avg_review_depth'] / 5)
    
    bars = axes[1,1].bar(range(len(df)), df['composite_score'], color=colors, alpha=0.7)
    axes[1,1].set_title('Quality Score (Lower = Better)')
    axes[1,1].set_xlabel('Repository')
    axes[1,1].set_ylabel('Composite Quality Score')
    axes[1,1].set_xticks(range(len(df)))
    axes[1,1].set_xticklabels(repo_names, rotation=45, ha='right')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='With Code Review'),
    Patch(facecolor='red', alpha=0.7, label='Without Code Review')]
    axes[1,1].legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('results/analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual detailed charts
    # Detailed bug resolution analysis
    if len(df_with_resolution) > 0:
        plt.figure(figsize=(12, 6))
        df_sorted = df_with_resolution.sort_values('avg_bug_resolution_days')
        bars = plt.bar(range(len(df_sorted)), df_sorted['avg_bug_resolution_days'], 
        color=['green' if x else 'red' for x in df_sorted['uses_code_review']])
        
        plt.title('Bug Resolution Time by Repository', fontsize=14, fontweight='bold')
        plt.xlabel('Repository')
        plt.ylabel('Average Resolution Time (days)')
        plt.xticks(range(len(df_sorted)), [repo.split('/')[-1] for repo in df_sorted['repo']], 
        rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
        f'{height:.1f}d', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/bug_resolution_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ Enhanced visualizations saved:")
    print("  - enhanced_code_review_analysis.png (comprehensive overview)")
    if len(df_with_resolution) > 0:
        print("  - bug_resolution_times.png (detailed resolution analysis)")
    
    # ----------------------------
    # STEP 7: Summary Report
    # ----------------------------
    print("\n" + "="*80)
    print("📝 EXECUTIVE SUMMARY")
    print("="*80)
    
    print("\n🎯 Three Key Metrics Analysis:")
    print(f"1. Bug Density: Projects with code review show {'lower' if with_review['bug_density'].mean() < without_review['bug_density'].mean() else 'higher'} bug density")
    print(f"2. Review Depth: Projects with code review have {with_review['avg_review_depth'].mean():.1f} comments/PR vs {without_review['avg_review_depth'].mean():.1f}")
    
    with_resolution_data = with_review[with_review['avg_bug_resolution_days'] > 0]
    without_resolution_data = without_review[without_review['avg_bug_resolution_days'] > 0]
    
    if len(with_resolution_data) > 0 and len(without_resolution_data) > 0:
        print(f"3. Bug Resolution: Projects with code review resolve bugs in {with_resolution_data['avg_bug_resolution_days'].mean():.1f} days vs {without_resolution_data['avg_bug_resolution_days'].mean():.1f} days")
    else:
        print("3. Bug Resolution: Insufficient data for comparison")
    
    print(f"\n✅ Analysis complete! Enhanced results saved to 'enhanced_code_review_analysis.csv'")
    
else:
    print("❌ No data collected. Please check your GitHub token and repository access.")

print(f"\n🎉 Enhanced script execution completed!")
print("📊 All three metrics analyzed: Bug Density, Review Depth, and Bug Resolution Time")