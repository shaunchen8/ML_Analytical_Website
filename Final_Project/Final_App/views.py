import os

from django.shortcuts import render
from django.http import Http404

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework import status
from django.conf import settings

from .models import FileModel
from .serializers import FileSerializer

def get_file_list():
    files=FileModel.objects.values_list('file_name',flat=True)
    return files.distinct()

def get_algorithm_list():
    return ['PimaDiabetesClassifier','WineQualityRegressor','NBAOutlier']

def get_algorithm(name):
    if name == 'PimaDiabetesClassifier':
        from .hw1_classification import run_algo1
        return run_algo1
    elif name == 'WineQualityRegressor':
        from .hw2_regression import run_algo2
        return run_algo2
    elif name == 'NBAOutlier':
        from .hw3_outlier import run_algo3
        return run_algo3
    else:
        # Http404 is imported from django.http
        raise Http404('<h1>Target algorithm not found</h1>')

def run_analytic(file_path, algo):
    file_abs_path = os.path.join(settings.MEDIA_ROOT, str(file_path))
    print(file_abs_path)
    return algo(file_abs_path)

class YourViewName(APIView):
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    renderer_classes = [TemplateHTMLRenderer]
    template_name = 'index.html'

    # See Django REST Request class here:
    # https://www.django-rest-framework.org/api-guide/requests/
    def get(self, request):
        return Response({'files': get_file_list(),'algorithms': get_algorithm_list()},status=status.HTTP_200_OK)
    
    def post(self, request):

        if 'upload' in request.data:
            file_serializer = FileSerializer(data=request.data)
            
            if file_serializer.is_valid():
                file_serializer.save()
                #return Response({'status': 'Upload successful!'},status=status.HTTP_201_CREATED)
                return Response({'files': get_file_list(),'algorithms': get_algorithm_list()},status=status.HTTP_200_OK)
                
        elif 'analytic' in request.data:
            # Run analytics on dataset as specified by file_name and
            # analytic_id received in the post request
            query_file_name = request.data['file_name']
            query_algorithm = request.data['algorithm']
            # Find file path to local folder
            file_obj = FileModel.objects.get(file_name=query_file_name)
            file_path = file_obj.file_content
            # Find algorithm
            algo_obj = get_algorithm(query_algorithm)
            analyticresult = run_analytic(file_path, algo_obj)
            return Response({'files':get_file_list(),'algorithms':get_algorithm_list(),'result_plot':analyticresult},status=status.HTTP_200_OK)
            #print(file_obj)
            #return Response(status=status.HTTP_200_OK)

        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        