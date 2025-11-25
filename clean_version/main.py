# main.py

from data_collector import collect_prompt_length_data, show_data_summary
from plot_generator import load_data, plot_combined_results, plot_comparison_across_tasks
import os
name_model = os.getenv('NAME_MODEL', 't5-small')

def main():
    """Главная функция - выбор режима работы"""
    
    print("🎯 Prompt Length Analysis System")
    print("1. Collect new data")
    print("2. Generate plots from existing data")
    print("3. Show data summary")
    
    choice = input("\nSelect mode (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting data collection...")
        collect_prompt_length_data(name_model=name_model)
        
    elif choice == "2":
        print("\n📊 Generating plots...")
        data = load_data()
        if data:
            plot_combined_results(data)
            plot_comparison_across_tasks(data)
        
    elif choice == "3":
        print("\n📋 Showing data summary...")
        show_data_summary()
        
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    print(f"name model = {name_model}")
    main()