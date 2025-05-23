from property.price import predict_price
import pandas as pd
import socket
from django.shortcuts import render, redirect, get_list_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
# from .forms import *
from .models import *
import uuid
from django.conf import settings
from django.core.mail import send_mail
from django.views.decorators.csrf import csrf_exempt
from datetime import date, timedelta
from django.db.models.functions import Now


def home(request):
    print("Welcome to home")
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(local_ip)

    agent_obj = Agent.objects.all().first()
    propertyArray = []
    property_obj = reversed(Property.objects.all())
    for i in property_obj:
        if i == 3:
            break
        ap = propertyArray.append(i)

    details = {
        "aname": agent_obj.name,
        'property': propertyArray
    }
    # print(agent['name'])
    return render(request, 'index.html', details, )


def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username').lower()
        email = request.POST.get('email')
        password = request.POST.get('password')
        password2 = request.POST.get('cpassword')

        try:
            if User.objects.filter(username=username).first():
                messages.success(request, 'Username Already Taken.')
                return redirect("sign-up")
            if User.objects.filter(email=email).first():
                messages.success(request, 'Email Already Taken.')
                return redirect('sign-up')

            if password == password2:
                user_obj = User.objects.create_user(username, email, password)
                user_obj.save()
                print("user_created")
                auth_token = str(uuid.uuid4())
                profile_obj = Profile(user=user_obj, auth_token=auth_token)
                profile_obj.save()
                send_email_after_registration(email, auth_token)

                return redirect('home')
        except Exception as e:
            print("Daily limit Exceed")
            return redirect('sign-in')
    return render(request, 'auth/sign-up.html')


def send_email_after_registration(email, token):
    subject = "Your accounts need to be verified!!"
    n1 = "\n"
    message = f'Hi click the link to verify your account http://127.0.0.1:8000/verify/{token}'
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email]
    send_mail(subject, message, email_from, recipient_list)


def send_email_for_password(email, token, uname):
    subject = "Your accounts need to be verified!!"
    n1 = "\n"
    message = f' Hello {uname}, \n Please click the link to reset your password \n http://127.0.0.1:8000/verifyforpassword/{token} \n Thanks for joining with us.\n Team Renvest'
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email]
    send_mail(subject, message, email_from, recipient_list)


def verify(request, auth_token):
    try:
        profile_obj = Profile.objects.filter(auth_token=auth_token).first()
        if profile_obj.is_verified:
            messages.add_message(request, messages.INFO,
                                 "E-mail is already Verified!")
            return redirect('login')
        if profile_obj:
            profile_obj.is_verified = True
            profile_obj.save()
            messages.add_message(request, messages.INFO,
                                 "E-mail Verified Successfull")
            return redirect('email-verify')
        else:
            return redirect('error')

    except Exception as e:
        print("EXCEPT")
        print(e)


def verifyforpassword(request, auth_token):
    try:
        forgot_obj = ForgotPassword.objects.filter(
            auth_token=auth_token).first()
        print(forgot_obj)
        if forgot_obj:
            forgot_obj.is_checked = True
            forgot_obj.save()
        if request.method == "POST":
            password = request.POST.get("password")
            cpassword = request.POST.get("cpassword")

            if password == cpassword:
                user_obj = User.objects.filter(email=forgot_obj.email).first()
                user_obj.set_password(password)
                user_obj.save()
                return redirect('sign-in')

    except Exception as e:
        print(e)
    return render(request, 'auth/reset-password.html')


def email_verify(request):
    return render(request, 'auth/email-verify.html')


def signin(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        username = User.objects.filter(email=email).first()
        user = authenticate(request, username=username, password=password)
        print(user)
        if user is not None:
            login(request, user)
            return redirect('home')
    return render(request, 'auth/sign-in.html')


def signout(request):
    logout(request)
    return redirect('home')

# def send_email_for_password(email, token):
#     subject = "Your accounts need to be verified!!"
#     n1 = "\n"
#     message = f'Hi click the link to verify your account http://127.0.0.1:8000//verify/{token}'
#     email_from = settings.EMAIL_HOST_USER
#     recipient_list = [email]
#     send_mail(subject, message, email_from, recipient_list)


def forgotPassword(request):
    if request.method == "POST":
        email = request.POST.get("email")

        user_obj = User.objects.filter(email=email).first()
        print(user_obj, type(user_obj))
        if user_obj is not None:
            ForgotPassword.objects.filter(email=email).all().delete()
            print("if")
            uname = user_obj
            auth_token = str(uuid.uuid4())
            forgot_obj = ForgotPassword(email=email, auth_token=auth_token)
            forgot_obj.save()
            send_email_for_password(email, auth_token, uname)
            return redirect('home')
    return render(request, 'auth/forgotPassword.html')


# def resetPassword(request):
#     if request.method == "POST":
#         password = request.POST.get('password')
#         cpassword = request.POST.get('cpassword')

#         if password == cpassword:
#             # forgot_obj = ForgotPassword.objects.filter(
#             #     auth_token=auth_token).first()
#             # if forgot_obj is not None:
#             #     user_obj = User.objects.filter(email=forgot_obj.email).first()
#             #     user_obj.password = password
#             #     user_obj.save()
#             #     return redirect('sign-in')
#             return redirect('sign-in')
#     else:
#         return render(request, 'auth/reset-password.html')

#     return render(request, 'auth/reset-password.html')


def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        contact_create = Contact(
            name=name, email=email, phone=phone, subject=subject, message=message)
        contact_create.save()

        return redirect(home)

    # fn = ContactForm
    data = {
        # 'form': fn,
    }
    return render(request, 'contact.html', data)


def about(request):
    return render(request, 'about.html')


def profile(request):
    return render(request, 'profile.html')


def agency_registeration(request):
    if request.user.is_authenticated:
        if request.method == "POST":
            agency_name = request.POST.get('agency_name')
            agency_tagline = request.POST.get('agency_tagline')
            agency_phone = request.POST.get('agency_phone')
            agency_email = request.POST.get('agency_email')
            agency_description = request.POST.get('agency_description')
            agency_facebook = request.POST.get('agency_facebook')
            agency_instagram = request.POST.get('agency_instagram')
            agency_image = request.FILES['agency_image']

            agency_create = Agency(user=request.user, agency_name=agency_name,
                                   agency_tagline=agency_tagline, agency_phone=agency_phone, agency_email=agency_email,
                                   agency_description=agency_description, agency_facebook=agency_facebook,
                                   agency_instagram=agency_instagram, agency_image=agency_image)
            agency_create.save()

            return redirect('home')
    else:
        return redirect('sign-in')

    return render(request, 'agency/agency-registeration.html')


def agency_list(request):
    agency_all = sorted(Agency.objects.all())
    print(agency_all)
    return render(request, 'agency/agency-list.html', {"agency_list": agency_all})


def agency_details(request, agency_name):
    agency_obj = Agency.objects.filter(agency_name=agency_name).first()
    return render(request, 'agency/agency-details.html', {'agency': agency_obj})

# def agency_details(request):
#     return render(request, 'agency-details.html')


def agent_registeration(request):
    if request.user.is_authenticated:
        if request.method == "POST":
            aname = request.POST.get('name')
            agency = request.POST.get('agency')
            phone = request.POST.get('phone')
            email = request.POST.get('email')
            description = request.POST.get('description')
            facebook = request.POST.get('facebook')
            instagram = request.POST.get('instagram')
            image = request.FILES['image']

            agent_create = Agent(user=request.user, name=aname, agency=agency, phone=phone, email=email,
                                 description=description, facebook=facebook, instagram=instagram, image=image)
            agent_create.save()
            print("Agent Create")
            return redirect('agent-list')
    else:
        return redirect('sign-in')

    return render(request, 'agent/agent-registeration.html')


def agent_list(request):
    agent_all = Agent.objects.all()
    count = agent_all.count()
    agent_group = []
    for i in agent_all:
        ap = agent_group.append(i)
    # agent_group
    # iterations for 6
    variables = {
        "count": count,
        "agent_list": agent_group
    }
    return render(request, 'agent/agent-list.html', variables)


def agent_details(request, username):
    agent_obj = Agent.objects.filter(user=username).first()
    return render(request, 'agent/agent-details.html', {'agent': agent_obj})


def property_listing(request):
    if request.user.is_authenticated:
        user_agency = Agency.objects.filter(user=request.user).first()
        print(user_agency)
        if user_agency:
            if request.method == "POST":
                agency_name = user_agency
                title = request.POST.get('title')
                address = request.POST.get('address')
                state = request.POST.get('state')
                city = request.POST.get('city')
                pincode = request.POST.get('pincode')
                property_type = request.POST.get('property_type')
                property_status = request.POST.get('property_status')
                property_price = request.POST.get('property_price')
                description = request.POST.get('description')
                main_pic = request.FILES['main_pic']
                pic_02 = request.FILES['pic_02']
                pic_03 = request.FILES['pic_03']

                property_create = Property(
                    agency_name=agency_name, title=title, address=address, state=state, city=city, pincode=pincode,
                    property_type=property_type, property_status=property_status,
                    property_price=property_price, description=description, main_pic=main_pic, pic_02=pic_02, pic_03=pic_03)
                property_create.save()
                return redirect('home')
    return render(request, 'property/property-listing.html')


def property_details(request, pk):
    property_obj = Property.objects.filter(id=pk).first()
    agency_obj = Agency.objects.filter(id=property_obj.agency_name.id).first()

    if request.method == "POST":
        date = request.POST.get('date')
        b = date.split("/")
        format = f"{b[2]}-{b[0]}-{b[1]}"
        time = request.POST.get('time')
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        message = request.POST.get('message')

        schedule_create = Schedule(
            date=format, time=time, name=name, email=email, phone=phone, message=message,
            property_title=property_obj.title, agency_name=agency_obj.agency_name)
        schedule_create.save()
        return redirect('property-details', property_obj.id)
    data = {
        "Property": property_obj,
        "Agency": agency_obj,
    }
    return render(request, 'property/property-details.html', data)


def property_list(request):
    properties = Property.objects.all()
    print(properties)
    return render(request, 'property/property-list.html', {"property": properties})


def mapview(request):
    """
    Handles the map view and predicts property prices based on user input.
    """
    try:
        # Load data and extract unique locations
        data = pd.read_csv(
            r'C:\Users\Yamraj\Desktop\renvest\renvest.in\property\Bengaluru_House_Data.csv')
        location_list = data['location'].unique()

        if request.method == "POST":
            location = request.POST.get('location')
            sqft = float(request.POST.get('sqft', 0))
            bhk = int(request.POST.get('bhk', 0))
            bath = int(request.POST.get('bath', 0))

            # Validate inputs
            if not location or sqft <= 0 or bhk <= 0 or bath <= 0:
                messages.error(
                    request, "Invalid input. Please provide valid details.")
                return render(request, "measurements.html", {"location": location_list})

            # Predict price
            predicted_price = int(predict_price(
                location, sqft, bath, bhk) * 10000)

            return render(request, "predicted_price.html", {"value": predicted_price})

        return render(request, "measurements.html", {"location": location_list})

    except FileNotFoundError:
        messages.error(
            request, "Data file not found. Please check the file path.")
        return render(request, "measurements.html", {"location": []})
    except Exception as e:
        print(f"Error in mapview: {e}")
        messages.error(
            request, "An error occurred while processing your request.")
        return render(request, "measurements.html", {"location": []})


def num2words(num):
    under_20 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
                'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['Twenty', 'Thirty', 'Forty', 'Fifty',
            'Sixty', 'Seventy', 'Eighty', 'Ninety']
    above_100 = {100: 'Hundred', 1000: 'Thousand',
                 100000: 'Lakhs', 10000000: 'Crores'}

    if num < 20:
        return under_20[num]

    if num < 100:
        return tens[(int)(num/10)-2] + ('' if num % 10 == 0 else ' ' + under_20[num % 10])

    # find the appropriate pivot - 'Million' in 3,603,550, or 'Thousand' in 603,550
    pivot = max([key for key in above_100.keys() if key <= num])

    return num2words((int)(num/pivot)) + ' ' + above_100[pivot] + ('' if num % pivot == 0 else ' ' + num2words(num % pivot))
