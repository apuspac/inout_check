from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect
from django.views import generic
from django.urls import reverse, reverse_lazy


import os
from .app import authc
from .form import TestForm, CreateUserForm, LoginForm
from .models import login_data, person
#from .models import person, login_data

class MainPage(generic.ListView):
    """
    login history
    できればメインページとは違うところへ
    """
    template_name = "face_login/index.html"
    context_object_name = "login_data"

    def get_queryset(self):
        return login_data.objects.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)    #super:親クラス(ここではTemplateView)のメソッドを実行
        context["welcome"] = "1"
        return context

class UserDetailView(generic.DetailView):
    model = person
    template_name = "face_login/person_detail.html"

class UserListView(generic.ListView):
    template_name = "face_login/list.html"
    model = person
    context_object_name = "member_list"

class UserCreateView(generic.CreateView):
    model = person
    form_class = CreateUserForm
    template_name = "face_login/create.html"
    success_url = reverse_lazy("face_login:top")

class UserUpdateView(generic.UpdateView):
    template_name = "face_login/create.html"
    model = person
    form_class = CreateUserForm
    success_url = reverse_lazy("face_login:list")

    # def form_valid(self, form):
    #     post = form.save(commit=False)
    #     post.updated_by = self.request.user
    #     post.updated_by =

class LoginFormView(generic.ListView):
    template_name = "face_login/login.html"
    model = person
    form_class = LoginForm
    context_object_name = "member_list"
    #success_url = reverse_lazy("face_login:list")

    # def form_valid(self, form):
    #     if "1" in form.name:
    #         print("OK2")
    #     print("OK")
    #     return super().form_valid(form)

    def post(self, request, *args, **kwargs):
        if "1" in request.POST:
            print("OK2")
            authc.prepare_anthentication()
        if "2" in request.POST:
            print("OK3")
            authc.learning()
        if "3" in request.POST:
            authc.file_Authentication("test1")
            authc.file_Authentication("test2")
            authc.file_Authentication("test3")
            authc.file_Authentication("gakko")
        return render(request, "face_login/login.html")


class UserStatusView(generic.ListView):
    """

    """
    template_name = "face_login/status.html"
    model = person
    form_class = LoginForm
    context_object_name = "member_list"

    def post(self, request, *args, **kwargs):
        #hito = person.objects.get(id=request.POST.get())
        hito = request.POST["LOGIN"]
        user_login_status = person.objects.all().order_by(hito)
        print(user_login_status.login_time)
        return render(request, "face_login/index.html")



# class LoginView(generic.FormView):
#     template_name = "face_login/login.html"
#     model = person
#     context_object_name = "member_list"

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)    #super:親クラス(ここではTemplateView)のメソッドを実行
#         context["result"] = "0"
#         return context

#     def post(self, request):
#         print("post")
#         if "1" in request.POST:
#             # context["result"] = "1"
#         return context

    # def get(self, request, *args):
    #     return render(request, self.template_name)



# def index(request):
#     index_dict = {
#         "welcome" : "1"
#     }
#     return render(request, "face_login/index.html", index_dict)

def login(request):
    login_dict = {
        "form" : TestForm(),
        "result_id" : "0",
        "result": "失敗",

    }

    if request.method == "POST":
        if "authc_button" in request.POST:
            login_dict["result_id"] = authc.authentication()
            if request.POST['id'] == login_dict["result_id"]:
                login_dict["result"] = "成功"
            else:
                login_dict["result"] = "失敗"


    return render(request, "face_login/login.html", login_dict)