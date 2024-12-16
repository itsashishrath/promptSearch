import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .search_engine import ResumeSearchEngine

@csrf_exempt
def search_resumes(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        top_k = int(request.POST.get('top_k', 10))

        search_engine = ResumeSearchEngine()
        results = search_engine.search(prompt, top_k)

        # Get the full file paths for the top results
        output_results = []
        for result in results:
            file_path = os.path.join('C:\\Users\\hp\\Desktop\\nicheby\\downloaded_resumes', result['filename'])
            output_results.append({
                'id': result['id'],
                'filename': result['filename'],
                'score': result['score'],
                'file_path': file_path
            })

        return JsonResponse({
            'status': 'success',
            'results': output_results,
            'total_results': len(output_results)
        })
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)