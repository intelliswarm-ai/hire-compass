#!/usr/bin/env python
"""
Master test runner for HR Resume Matcher
Runs all test suites and generates comprehensive reports
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
import subprocess
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.unit.test_ranking_validation import run_ranking_tests
from tests.integration.test_full_pipeline import run_integration_tests
from tests.performance.test_large_scale_matching import LargeScaleMatchingTest


class TestReportGenerator:
    """Generate comprehensive test reports with visualizations"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start test execution"""
        self.start_time = time.time()
        print("=" * 80)
        print("HR RESUME MATCHER - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    def add_result(self, test_name: str, result: Dict[str, Any]):
        """Add test result"""
        self.results[test_name] = result
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate final test report"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Calculate overall statistics
        total_tests = sum(r.get("total_tests", 0) for r in self.results.values())
        total_failures = sum(r.get("failures", 0) for r in self.results.values())
        total_errors = sum(r.get("errors", 0) for r in self.results.values())
        
        report = {
            "test_suite": "HR Resume Matcher Comprehensive Tests",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(total_duration, 2),
            "overall_statistics": {
                "total_tests": total_tests,
                "total_passed": total_tests - total_failures - total_errors,
                "total_failures": total_failures,
                "total_errors": total_errors,
                "success_rate": round(((total_tests - total_failures - total_errors) / total_tests * 100) 
                                    if total_tests > 0 else 0, 2)
            },
            "test_results": self.results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cwd": os.getcwd()
            }
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        stats = report["overall_statistics"]
        print(f"\nTotal Tests Run: {stats['total_tests']}")
        print(f"Passed: {stats['total_passed']} ✅")
        print(f"Failed: {stats['total_failures']} ❌")
        print(f"Errors: {stats['total_errors']} 🔥")
        print(f"Success Rate: {stats['success_rate']}%")
        print(f"Total Duration: {report['duration_seconds']}s")
        
        print("\n" + "-" * 40)
        print("DETAILED RESULTS BY TEST SUITE:")
        print("-" * 40)
        
        for test_name, result in report["test_results"].items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            print(f"\n{test_name}: {status}")
            
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                print(f"  - Processing Time: {metrics.get('total_time_seconds', 'N/A')}s")
                print(f"  - Memory Used: {metrics.get('memory_used_mb', 'N/A')} MB")
                print(f"  - Success Rate: {metrics.get('success_rate', 'N/A')}%")
            
            if test_name == "ranking_validation":
                if "accuracy_percentage" in result:
                    print(f"  - Ranking Accuracy: {result['accuracy_percentage']}%")
            
            if test_name == "performance_test":
                test_results = result.get("test_results", {})
                if "single_resume_test" in test_results:
                    single = test_results["single_resume_test"]
                    print(f"  - Single Resume vs 300 Positions: {single.get('total_time', 'N/A')}s")
                
                if "ranking_validation" in test_results:
                    ranking = test_results["ranking_validation"]
                    print(f"  - Ranking Accuracy: {ranking.get('accuracy_percentage', 'N/A')}%")
        
        print("\n" + "=" * 80)
    
    def save_html_report(self, report: Dict[str, Any], filepath: str):
        """Generate HTML report with visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HR Resume Matcher Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px; flex: 1; margin: 0 10px; }}
        .stat-number {{ font-size: 36px; font-weight: bold; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .test-passed {{ background-color: #d4edda; }}
        .test-failed {{ background-color: #f8d7da; }}
        .performance-chart {{ margin: 20px 0; }}
        .chart-bar {{ height: 30px; background-color: #007bff; margin: 5px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HR Resume Matcher Test Report</h1>
        <p style="text-align: center; color: #666;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overall Statistics</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{report['overall_statistics']['total_tests']}</div>
                <div>Total Tests</div>
            </div>
            <div class="stat-box">
                <div class="stat-number passed">{report['overall_statistics']['total_passed']}</div>
                <div>Passed</div>
            </div>
            <div class="stat-box">
                <div class="stat-number failed">{report['overall_statistics']['total_failures']}</div>
                <div>Failed</div>
            </div>
            <div class="stat-box">
                <div class="stat-number {'passed' if report['overall_statistics']['success_rate'] > 80 else 'warning'}">{report['overall_statistics']['success_rate']}%</div>
                <div>Success Rate</div>
            </div>
        </div>
        
        <h2>Test Suite Results</h2>
        <table>
            <tr>
                <th>Test Suite</th>
                <th>Status</th>
                <th>Tests Run</th>
                <th>Duration</th>
                <th>Key Metrics</th>
            </tr>
"""
        
        for test_name, result in report["test_results"].items():
            status = "PASSED" if result.get("success", False) else "FAILED"
            status_class = "test-passed" if result.get("success", False) else "test-failed"
            
            # Extract key metrics based on test type
            key_metrics = []
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                key_metrics.append(f"Memory: {metrics.get('memory_used_mb', 'N/A')} MB")
                key_metrics.append(f"Avg Time: {metrics.get('average_processing_time', 'N/A')}s")
            
            if test_name == "performance_test":
                test_results = result.get("test_results", {})
                if "ranking_validation" in test_results:
                    ranking = test_results["ranking_validation"]
                    key_metrics.append(f"Ranking Accuracy: {ranking.get('accuracy_percentage', 'N/A')}%")
            
            html_content += f"""
            <tr class="{status_class}">
                <td>{test_name.replace('_', ' ').title()}</td>
                <td>{status}</td>
                <td>{result.get('total_tests', 'N/A')}</td>
                <td>{result.get('duration', 'N/A')}s</td>
                <td>{', '.join(key_metrics) if key_metrics else 'N/A'}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>Performance Metrics</h2>
        <div class="performance-chart">
"""
        
        # Add performance visualization if available
        if "performance_test" in report["test_results"]:
            perf_data = report["test_results"]["performance_test"]
            if "test_results" in perf_data:
                test_results = perf_data["test_results"]
                
                if "single_resume_test" in test_results:
                    single = test_results["single_resume_test"]
                    html_content += f"""
            <h3>Single Resume vs 300 Positions Performance</h3>
            <p>Vector Search Time: {single.get('vector_search_time', 'N/A')}s</p>
            <div class="chart-bar" style="width: {min(single.get('vector_search_time', 0) * 100, 100)}%;"></div>
            <p>Matching Time: {single.get('matching_time', 'N/A')}s</p>
            <div class="chart-bar" style="width: {min(single.get('matching_time', 0) * 50, 100)}%;"></div>
"""
        
        html_content += """
        </div>
        
        <h2>Test Execution Details</h2>
        <ul>
            <li>Total Duration: """ + str(report['duration_seconds']) + """s</li>
            <li>Python Version: """ + report['environment']['python_version'].split()[0] + """</li>
            <li>Platform: """ + report['environment']['platform'] + """</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"\n📊 HTML report saved to: {filepath}")


async def run_comprehensive_tests():
    """Run all test suites"""
    report_generator = TestReportGenerator()
    report_generator.start()
    
    # 1. Run Unit Tests (Ranking Validation)
    print("\n🧪 Running Unit Tests - Ranking Validation...")
    try:
        ranking_success = run_ranking_tests()
        
        # Load the generated report
        with open("tests/reports/ranking_validation_report.json", 'r') as f:
            ranking_report = json.load(f)
        
        ranking_report["success"] = ranking_success
        report_generator.add_result("ranking_validation", ranking_report)
    except Exception as e:
        print(f"❌ Ranking tests failed: {e}")
        report_generator.add_result("ranking_validation", {
            "success": False,
            "error": str(e),
            "total_tests": 0
        })
    
    # 2. Run Integration Tests
    print("\n🧪 Running Integration Tests...")
    try:
        integration_success = run_integration_tests()
        
        # Load the generated report
        with open("tests/reports/integration_test_report.json", 'r') as f:
            integration_report = json.load(f)
        
        integration_report["success"] = integration_success
        report_generator.add_result("integration_tests", integration_report)
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        report_generator.add_result("integration_tests", {
            "success": False,
            "error": str(e),
            "total_tests": 0
        })
    
    # 3. Run Performance Tests (Large Scale Matching)
    print("\n🧪 Running Performance Tests - Large Scale Matching...")
    try:
        perf_test = LargeScaleMatchingTest()
        perf_report = await perf_test.run_all_tests()
        perf_report["success"] = True
        report_generator.add_result("performance_test", perf_report)
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        report_generator.add_result("performance_test", {
            "success": False,
            "error": str(e),
            "total_tests": 0
        })
    
    # Generate final report
    final_report = report_generator.generate_report()
    
    # Save comprehensive report
    report_path = f"tests/reports/comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Generate HTML report
    html_path = report_path.replace('.json', '.html')
    report_generator.save_html_report(final_report, html_path)
    
    # Print summary
    report_generator.print_summary(final_report)
    
    # Save latest report link
    latest_path = "tests/reports/latest_report.json"
    with open(latest_path, 'w') as f:
        json.dump({
            "latest_report": report_path,
            "latest_html": html_path,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Comprehensive test report saved to:")
    print(f"   JSON: {report_path}")
    print(f"   HTML: {html_path}")
    
    # Return overall success
    return final_report["overall_statistics"]["success_rate"] >= 80


def main():
    """Main entry point"""
    print("🚀 Starting HR Resume Matcher Test Suite")
    print("This will test matching with 300 positions and validate ranking accuracy\n")
    
    # Ensure test data exists
    if not os.path.exists("tests/data/job_positions_300.json"):
        print("📝 Generating test data...")
        from tests.data.test_data_generator import generate_test_data
        generate_test_data()
    
    # Run all tests
    success = asyncio.run(run_comprehensive_tests())
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the reports for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()