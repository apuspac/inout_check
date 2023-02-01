from django.urls import path
from . import views

app_name = 'face_login'
urlpatterns = [
    path('top/', views.MainPage.as_view(), name="top"),
    path('status/', views.UserStatusView.as_view(), name='status'),
    path("login/", views.LoginFormView.as_view(), name="login"),
    path("list/", views.UserListView.as_view(), name="list"),
    path('detail/<int:pk>', views.UserDetailView.as_view()),
    path("create/", views.UserCreateView.as_view(), name = "create"),
    path("update/<int:pk>", views.UserUpdateView.as_view(), name = "update"),
]