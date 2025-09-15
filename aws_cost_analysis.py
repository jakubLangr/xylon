#!/usr/bin/env python3
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
from typing import Dict, List, Any

class AWSELBCostAnalyzer:
    def __init__(self):
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
    def run_aws_command(self, command: List[str]) -> Dict[Any, Any]:
        """Execute AWS CLI command and return JSON response"""
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"AWS CLI error: {e.stderr}")
            return {}
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {}

    def get_elb_costs_by_usage_type(self) -> pd.DataFrame:
        """Get ELB costs broken down by usage type"""
        command = [
            'aws', 'ce', 'get-cost-and-usage',
            '--time-period', f'Start={self.start_date},End={self.end_date}',
            '--granularity', 'DAILY',
            '--metrics', 'UnblendedCost',
            '--group-by', 'Type=DIMENSION,Key=USAGE_TYPE',
            '--filter', '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Load Balancing"]}}'
        ]
        
        data = self.run_aws_command(command)
        rows = []
        
        for result in data.get('ResultsByTime', []):
            date = result['TimePeriod']['Start']
            for group in result.get('Groups', []):
                usage_type = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                if cost > 0:  # Only include non-zero costs
                    rows.append({'Date': date, 'UsageType': usage_type, 'Cost': cost})
        
        return pd.DataFrame(rows)

    def get_elb_costs_by_region(self) -> pd.DataFrame:
        """Get ELB costs broken down by region"""
        command = [
            'aws', 'ce', 'get-cost-and-usage',
            '--time-period', f'Start={self.start_date},End={self.end_date}',
            '--granularity', 'DAILY',
            '--metrics', 'UnblendedCost',
            '--group-by', 'Type=DIMENSION,Key=REGION',
            '--filter', '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Load Balancing"]}}'
        ]
        
        data = self.run_aws_command(command)
        rows = []
        
        for result in data.get('ResultsByTime', []):
            date = result['TimePeriod']['Start']
            for group in result.get('Groups', []):
                region = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                if cost > 0:
                    rows.append({'Date': date, 'Region': region, 'Cost': cost})
        
        return pd.DataFrame(rows)

    def get_elb_list(self) -> List[Dict]:
        """Get list of all ELBs with cost estimation"""
        # Get ALBs
        alb_command = ['aws', 'elbv2', 'describe-load-balancers']
        alb_data = self.run_aws_command(alb_command)
        
        # Get Classic ELBs
        elb_command = ['aws', 'elb', 'describe-load-balancers']
        elb_data = self.run_aws_command(elb_command)
        
        elbs = []
        
        # Process ALBs/NLBs
        for lb in alb_data.get('LoadBalancers', []):
            # Calculate rough hourly cost
            hourly_cost = 0.0225 if lb['Type'] == 'application' else 0.045  # ALB vs NLB base cost
            daily_cost = hourly_cost * 24
            
            elbs.append({
                'Name': lb['LoadBalancerName'],
                'Type': lb['Type'],
                'ARN': lb['LoadBalancerArn'],
                'State': lb['State']['Code'],
                'Region': lb['AvailabilityZones'][0]['ZoneName'][:-1],
                'EstimatedDailyCost': daily_cost,
                'AZCount': len(lb['AvailabilityZones'])
            })
        
        # Process Classic ELBs
        for lb in elb_data.get('LoadBalancerDescriptions', []):
            daily_cost = 0.025 * 24  # Classic ELB hourly cost
            
            elbs.append({
                'Name': lb['LoadBalancerName'],
                'Type': 'classic',
                'ARN': f"classic-elb-{lb['LoadBalancerName']}",
                'State': 'active',
                'Region': lb['AvailabilityZones'][0][:-1] if lb['AvailabilityZones'] else 'unknown',
                'EstimatedDailyCost': daily_cost,
                'AZCount': len(lb['AvailabilityZones'])
            })
        
        return elbs

    def get_cloudwatch_metrics_for_elb(self, elb: Dict) -> Dict:
        """Get CloudWatch metrics for specific ELB"""
        if elb['Type'] == 'application':
            namespace = 'AWS/ApplicationELB'
            metric_names = ['ProcessedBytes', 'NewConnectionCount', 'ActiveConnectionCount', 'ConsumedLCUs']
            lb_dimension = '/'.join(elb['ARN'].split('/')[-3:]) if '/' in elb['ARN'] else elb['Name']
        elif elb['Type'] == 'network':
            namespace = 'AWS/NetworkELB'
            metric_names = ['ProcessedBytes_TCP', 'NewFlowCount_TCP', 'ActiveFlowCount_TCP', 'ConsumedLCUs']
            lb_dimension = '/'.join(elb['ARN'].split('/')[-3:]) if '/' in elb['ARN'] else elb['Name']
        else:
            namespace = 'AWS/ELB'
            metric_names = ['RequestCount', 'HTTPCode_Backend_2XX', 'Latency']
            lb_dimension = elb['Name']
        
        metrics = {'ELB': elb['Name'], 'Type': elb['Type'], 'Region': elb['Region']}
        
        for metric_name in metric_names:
            command = [
                'aws', 'cloudwatch', 'get-metric-statistics',
                '--namespace', namespace,
                '--metric-name', metric_name,
                '--dimensions', f'Name=LoadBalancer,Value={lb_dimension}',
                '--statistics', 'Sum,Average,Maximum',
                '--start-time', f"{self.start_date}T00:00:00Z",
                '--end-time', f"{self.end_date}T00:00:00Z",
                '--period', '86400',
                '--region', elb['Region']
            ]
            
            data = self.run_aws_command(command)
            datapoints = data.get('Datapoints', [])
            if datapoints:
                metrics[f'{metric_name}_Total'] = sum(dp.get('Sum', 0) for dp in datapoints)
                metrics[f'{metric_name}_Avg'] = sum(dp.get('Average', 0) for dp in datapoints) / len(datapoints)
                metrics[f'{metric_name}_Max'] = max(dp.get('Maximum', 0) for dp in datapoints)
            else:
                metrics[f'{metric_name}_Total'] = 0
                metrics[f'{metric_name}_Avg'] = 0
                metrics[f'{metric_name}_Max'] = 0
        
        return metrics

    def create_visualizations(self, usage_df: pd.DataFrame, region_df: pd.DataFrame, elbs: List[Dict]):
        """Create cost visualization charts"""
        fig = plt.figure(figsize=(20, 15))
        
        # Create subplots with different sizes
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        if not usage_df.empty:
            # 1. Cost by Usage Type (Pie Chart)
            ax1 = fig.add_subplot(gs[0, 0])
            usage_total = usage_df.groupby('UsageType')['Cost'].sum().sort_values(ascending=False)
            ax1.pie(usage_total.values, labels=[label.replace('-', '\n') for label in usage_total.index], 
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('ELB Costs by Usage Type')
            
            # 2. Cost by Usage Type (Bar Chart)
            ax2 = fig.add_subplot(gs[0, 1])
            usage_total.plot(kind='bar', ax=ax2)
            ax2.set_title('ELB Costs by Usage Type ($)')
            ax2.set_ylabel('Cost ($)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Daily Cost Trend
            ax3 = fig.add_subplot(gs[0, 2])
            daily_costs = usage_df.groupby('Date')['Cost'].sum()
            daily_costs.plot(kind='line', ax=ax3, marker='o')
            ax3.set_title('Daily ELB Cost Trend')
            ax3.set_ylabel('Cost ($)')
            ax3.tick_params(axis='x', rotation=45)
        
        if not region_df.empty:
            # 4. Cost by Region
            ax4 = fig.add_subplot(gs[1, 0])
            region_total = region_df.groupby('Region')['Cost'].sum().sort_values(ascending=False)
            region_total.plot(kind='bar', ax=ax4)
            ax4.set_title('ELB Costs by Region ($)')
            ax4.set_ylabel('Cost ($)')
            ax4.tick_params(axis='x', rotation=45)
        
        if elbs:
            # 5. ELB Count by Type
            ax5 = fig.add_subplot(gs[1, 1])
            elb_types = pd.Series([elb['Type'] for elb in elbs]).value_counts()
            ax5.pie(elb_types.values, labels=elb_types.index, autopct='%1.0f')
            ax5.set_title('ELB Count by Type')
            
            # 6. ELB Count by Region
            ax6 = fig.add_subplot(gs[1, 2])
            elb_regions = pd.Series([elb['Region'] for elb in elbs]).value_counts()
            elb_regions.plot(kind='bar', ax=ax6)
            ax6.set_title('ELB Count by Region')
            ax6.tick_params(axis='x', rotation=45)
            
            # 7. Estimated vs Actual Costs (if we have usage data)
            if not usage_df.empty:
                ax7 = fig.add_subplot(gs[2, :])
                estimated_cost = sum(elb['EstimatedDailyCost'] for elb in elbs) * 7
                actual_cost = usage_df['Cost'].sum()
                
                costs = pd.Series({'Estimated Base Cost': estimated_cost, 'Actual Total Cost': actual_cost})
                costs.plot(kind='bar', ax=ax7)
                ax7.set_title('Estimated vs Actual Weekly Costs')
                ax7.set_ylabel('Cost ($)')
                
                # Add difference annotation
                difference = actual_cost - estimated_cost
                ax7.annotate(f'Difference: ${difference:.2f}\n({difference/estimated_cost*100:.1f}% over base)', 
                           xy=(0.5, max(costs) * 0.8), ha='center')
        
        plt.savefig('elb_cost_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualization saved as 'elb_cost_analysis.png'")

    def generate_actionable_insights(self, usage_df: pd.DataFrame, region_df: pd.DataFrame, elbs: List[Dict]):
        """Generate actionable recommendations"""
        print("\n" + "="*80)
        print("🎯 ACTIONABLE ELB COST OPTIMIZATION INSIGHTS")
        print("="*80)
        
        # Basic inventory analysis
        elb_count = len(elbs)
        elb_types = {}
        for elb in elbs:
            elb_types[elb['Type']] = elb_types.get(elb['Type'], 0) + 1
        
        print(f"📊 ELB INVENTORY:")
        print(f"   • Total ELBs: {elb_count}")
        for elb_type, count in elb_types.items():
            print(f"   • {elb_type.title()}: {count}")
        
        if not usage_df.empty:
            # Cost analysis
            total_cost = usage_df['Cost'].sum()
            estimated_base_cost = sum(elb['EstimatedDailyCost'] for elb in elbs) * 7
            
            print(f"\n💰 COST ANALYSIS (7 days):")
            print(f"   • Total actual cost: ${total_cost:.2f}")
            print(f"   • Estimated base cost: ${estimated_base_cost:.2f}")
            print(f"   • Extra cost (LCUs/data): ${total_cost - estimated_base_cost:.2f}")
            print(f"   • Monthly projection: ${total_cost * 4.3:.2f}")
            
            # Usage breakdown
            print(f"\n📈 COST BREAKDOWN:")
            usage_breakdown = usage_df.groupby('UsageType')['Cost'].sum().sort_values(ascending=False)
            for usage_type, cost in usage_breakdown.items():
                percentage = (cost / total_cost) * 100
                print(f"   • {usage_type}: ${cost:.2f} ({percentage:.1f}%)")
        
        if not region_df.empty:
            print(f"\n🌍 REGIONAL DISTRIBUTION:")
            region_breakdown = region_df.groupby('Region')['Cost'].sum().sort_values(ascending=False)
            for region, cost in region_breakdown.items():
                print(f"   • {region}: ${cost:.2f}")
        
        # Generate recommendations
        print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS:")
        
        # Check for too many ELBs
        if elb_count > 5:
            print(f"   ⚠️  HIGH ELB COUNT ({elb_count} total)")
            print(f"      → Review if ELBs can be consolidated")
            print(f"      → Consider path-based routing on single ALB")
        
        # Check ELB types
        if 'classic' in elb_types and elb_types['classic'] > 0:
            print(f"   ⚠️  CLASSIC ELBs DETECTED ({elb_types['classic']} found)")
            print(f"      → Migrate to ALB/NLB for cost savings and features")
            print(f"      → Classic ELBs are more expensive per hour")
        
        if not usage_df.empty:
            # Check for high LCU usage
            lcu_costs = usage_breakdown[usage_breakdown.index.str.contains('LCU', case=False, na=False)]
            if not lcu_costs.empty and lcu_costs.sum() > total_cost * 0.4:
                print(f"   ⚠️  HIGH LCU USAGE (${lcu_costs.sum():.2f}, {lcu_costs.sum()/total_cost*100:.1f}%)")
                print(f"      → Optimize connection patterns")
                print(f"      → Review rule complexity")
                print(f"      → Consider connection pooling")
            
            # Check for data transfer costs
            transfer_keywords = ['Transfer', 'DataTransfer', 'Data-Transfer', 'Regional', 'Out']
            transfer_costs = usage_breakdown[
                usage_breakdown.index.str.contains('|'.join(transfer_keywords), case=False, na=False)
            ]
            if not transfer_costs.empty and transfer_costs.sum() > total_cost * 0.2:
                print(f"   ⚠️  HIGH DATA TRANSFER (${transfer_costs.sum():.2f})")
                print(f"      → Review cross-AZ traffic patterns")
                print(f"      → Optimize target placement")
        
        # Multi-AZ recommendations
        multi_az_elbs = [elb for elb in elbs if elb['AZCount'] > 2]
        if multi_az_elbs:
            print(f"   💡 MULTI-AZ OPTIMIZATION ({len(multi_az_elbs)} ELBs in 3+ AZs)")
            print(f"      → Review if all AZs are needed")
            print(f"      → Each additional AZ increases costs")
        
        print(f"\n🔧 IMMEDIATE ACTIONS:")
        print(f"   1. Audit unused/low-traffic ELBs")
        print(f"   2. Migrate Classic ELBs to ALB")
        print(f"   3. Review target group health checks")
        print(f"   4. Analyze CloudWatch metrics for top ELBs")
        
        # Generate specific commands
        print(f"\n📋 INVESTIGATION COMMANDS:")
        if elbs:
            top_elb = elbs[0]
            print(f"   # Check metrics for {top_elb['Name']}:")
            if top_elb['Type'] == 'application':
                print(f"   aws cloudwatch get-metric-statistics \\")
                print(f"     --namespace AWS/ApplicationELB \\")
                print(f"     --metric-name ConsumedLCUs \\")
                print(f"     --dimensions Name=LoadBalancer,Value={top_elb['Name']} \\")
                print(f"     --start-time {self.start_date}T00:00:00Z \\")
                print(f"     --end-time {self.end_date}T00:00:00Z \\")
                print(f"     --period 3600 --statistics Average,Maximum \\")
                print(f"     --region {top_elb['Region']}")
        
        print(f"\n   # Get detailed cost breakdown:")
        print(f"   aws ce get-cost-and-usage \\")
        print(f"     --time-period Start={self.start_date},End={self.end_date} \\")
        print(f"     --granularity DAILY --metrics UnblendedCost,UsageQuantity \\")
        print(f"     --group-by Type=DIMENSION,Key=USAGE_TYPE \\")
        print(f"     --filter '{{\"Dimensions\":{{\"Key\":\"SERVICE\",\"Values\":[\"Amazon Elastic Load Balancing\"]}}}}'")

    def run_analysis(self):
        """Run complete ELB cost analysis"""
        print("🔍 Starting ELB Cost Analysis...")
        
        # Get cost data
        print("   📊 Fetching usage type costs...")
        usage_df = self.get_elb_costs_by_usage_type()
        
        print("   🌍 Fetching regional costs...")
        region_df = self.get_elb_costs_by_region()
        
        print("   📋 Getting ELB inventory...")
        elbs = self.get_elb_list()
        
        if usage_df.empty and region_df.empty and not elbs:
            print("❌ No ELB data found. Check your AWS credentials and permissions.")
            return
        
        # Create visualizations
        print("   📈 Creating visualizations...")
        self.create_visualizations(usage_df, region_df, elbs)
        
        # Generate insights
        self.generate_actionable_insights(usage_df, region_df, elbs)
        
        # Save raw data
        if not usage_df.empty:
            usage_df.to_csv('elb_usage_costs.csv', index=False)
            print(f"\n💾 Usage costs saved to: elb_usage_costs.csv")
        
        if not region_df.empty:
            region_df.to_csv('elb_regional_costs.csv', index=False)
            print(f"💾 Regional costs saved to: elb_regional_costs.csv")
        
        if elbs:
            elb_df = pd.DataFrame(elbs)
            elb_df.to_csv('elb_inventory.csv', index=False)
            print(f"💾 ELB inventory saved to: elb_inventory.csv")

if __name__ == "__main__":
    analyzer = AWSELBCostAnalyzer()
    analyzer.run_analysis()