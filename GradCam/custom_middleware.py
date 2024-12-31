from django.utils.deprecation import MiddlewareMixin

class AllowIframeMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        if request.path.startswith('/home'):  # Allow embedding only for '/home'
            response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Allow same-origin frames
        return response
