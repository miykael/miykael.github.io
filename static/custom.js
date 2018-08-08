(function($){
	'use-strict';



	/* ---------------------------------------------
	 Initializing Functions Triggered On Window Load
	 --------------------------------------------- */
	$(window).on('load', function(){
		initPreloaderFade();
	});


	/* ---------------------------------------------
	 Initializing Functions Triggered On Window Resize
	 --------------------------------------------- */
	$(window).on('resize', function(){
		initHeroHeight();
		initPortfolioResize();
		initContactMapHeight();
	});


	/* ---------------------------------------------
	 Initializing Functions Triggered On Window Scroll
	 --------------------------------------------- */
	$(window).on('scroll', function(){
		initHeaderAnimation();
		initSearchAreaHide();
	});


	/* ---------------------------------------------
	 Initializing All Functions
	 --------------------------------------------- */
	initColorScheme();
	initHeroHeight();
	initParallax();
	initSectionHighlight();
	initImageBackground();
	initSearchArea();
	initAnimateScroll();
	initNavMenu();
	initMagnificPopup();
	initPortfolio();
	initMasonry();
	initProgressBar();
	initCarousel();
	initWowAnimation();
	initMiscellaneous();
	initContactForm();
	initContactMapHeight();
	initGoogleMap();



	/* ---------------------------------------------
	 Color Scheme Select
	 --------------------------------------------- */
	function initColorScheme(){
		$('.color-scheme-select-btn').on('click', function(){
			$('.color-scheme-select').toggleClass('color-scheme-select-visible');
		});

		$('.color-scheme-content').on('click', function(){
			var colorLink = 'css/' + $(this).attr('id') + '.css';
			$('#tile-colorscheme-css').attr('href', colorLink);
		});
	}


	/* ---------------------------------------------
	 Preloader Fadeout
	 --------------------------------------------- */
	function initPreloaderFade(){
		$('.preloader').fadeOut();
	}


	/* ---------------------------------------------
	 Section Highlighting
	 --------------------------------------------- */
	function initSectionHighlight(){
		$('section').each(function(){
			$(this).waypoint(function(direction){
				if(direction === 'down'){
					var sec_id = $(this).attr('id');
					var current_section_link = '.' + sec_id + '-nav';
					$('.main-nav > ul > li > a').removeClass('active-nav');
					$(current_section_link).addClass('active-nav');
				}
			} , {offset: '130px'});
			$(this).waypoint(function(direction){
				if(direction === 'up'){
					var sec_id = $(this).attr('id');
					var current_section_link = '.' + sec_id + '-nav';
					$('.main-nav > ul > li > a').removeClass('active-nav');
					$(current_section_link).addClass('active-nav');
				}
			} , {offset: function(){return -$(this).height() + 130;}});
		});
	}


	/* ---------------------------------------------
	 Hero Height
	 --------------------------------------------- */
	function initHeroHeight(){
		$('.hero-height').height($(window).height());
	}


	/* ---------------------------------------------
	 Parallax Background
	 --------------------------------------------- */
	function initParallax(){
		if(!device.tablet() && !device.mobile()){
			$('.parallax-section').each(function(){
				$('.parallax-section').parallaxScroll("50%", 0.3);
			});
			$('.parallax-layer').parallax();
		}
	}


	/* ---------------------------------------------
	 Image Background
	 --------------------------------------------- */
	function initImageBackground(){
		$(".image-bg").each(function(){
			if($(this).attr("data-image-bg")){
				$(this).css({
					"background": "url(" + $(this).data("image-bg") + ")",
					'background-position': 'center top',
					'background-repeat': 'no-repeat',
					'background-size': '100%'
				});
			}
		});
	}


	/* ---------------------------------------------
	 Search Area
	 --------------------------------------------- */
	function initSearchArea(){
		$('.search-open').on('click', function(){
			if($('.search-area').hasClass('search-area-hidden')){
				$('.search-area').removeClass('search-area-hidden');
				$('.search-area').addClass('search-area-visible');
			}
		});

		$('html').on('click', function(){
			if($('.search-area').hasClass('search-area-visible')){
				$('.search-area').removeClass('search-area-visible');
				$('.search-area').addClass('search-area-hidden');
			}
		});

		$('.search-area, .search-open').on('click', function(e){
			e.stopPropagation();
		});
	}


	/* ---------------------------------------------
	 Search Area Hide On Scroll
	 --------------------------------------------- */
	function initSearchAreaHide(){
		if($(window).scrollTop() < 100){
			if($('.search-area').hasClass('search-area-visible')){
				$('.search-area').removeClass('search-area-visible');
				$('.search-area').addClass('search-area-hidden');
			}
		}
	}


	/* ---------------------------------------------
	 Animate Scroll
	 --------------------------------------------- */
	function initAnimateScroll(){
		$('.animatescroll-link').on('click', function(e){
			e.preventDefault();
		});
	}


	/* ---------------------------------------------
	 Navigation Menu
	 --------------------------------------------- */
	function initNavMenu(){
		$('.menu-bar').on('click', function(){
			if($('.main-nav').hasClass('hidden')){
				$('.main-nav').removeClass('hidden');
				$('.main-nav').addClass('active');
			}
		});

		$('.menu-close').on('click', function(){
			if($('.main-nav').hasClass('active')){
				$('.main-nav').removeClass('active');
				$('.main-nav').addClass('hidden');
			}
		});
	}


	/* ---------------------------------------------
	 Header Animation
	 --------------------------------------------- */
	function initHeaderAnimation(){
		var scroll_top = $(document).scrollTop();
		if(scroll_top >= 100){
			$('.header').removeClass('header-hidden');
			$('.header').addClass('header-visible');
		}else{
			$('.header').removeClass('header-visible');
			$('.header').addClass('header-hidden');
		}
	}


	/* ---------------------------------------------
	 Magnific Popup Plugin
	 --------------------------------------------- */
	function initMagnificPopup(){
		$('.portrait-mfp').magnificPopup({
			type: 'image',
			gallery: {
				enabled: true
			}
		});

		$('.portfolio-mfp').magnificPopup({
			type: 'image',
			gallery: {
				enabled: true
			}
		});

		$(".video-mfp").magnificPopup({
			type: 'iframe',
			gallery: {
				enabled: true
			}
		});
	}


	/* ---------------------------------------------
	 Masonry
	 --------------------------------------------- */
	function initMasonry(){
		$(".masonry").imagesLoaded(function(){
			$(".masonry").masonry();
		});
	}


	/* ---------------------------------------------
	 Portfolio
	 --------------------------------------------- */
	function initPortfolio(){
		$('.portfolio-items').imagesLoaded(function(){
			$('.portfolio-items').show();
			$('.portfolio-items').isotope({
				filter:'*',
				layoutMode:'masonry',
				animationOptions:{
					duration:750,
					easing:'linear'
				}
			});
		});

		$('.filter').find('a').on('click', function(){
			$('.portfolio-items').isotope({
				filter	: $(this).attr('data-filter'),
				animationOptions: {
					duration: 750,
					queue: false
				}
			});
			return false;
		});

		$('.filter a').on('click', function(){
			if (!$(this).hasClass('active')){
				$('.filter a').removeClass('active');
				$(this).addClass('active');
			}
		});
	}


	/* ---------------------------------------------
	 Portfolio Resize
	 --------------------------------------------- */
	function initPortfolioResize(){
		$('.portfolio-items').isotope({
			filter	: $('.filter').find('a.active').attr('data-filter'),
			animationOptions: {
				duration: 750,
				queue: false
			}
		});
		return false;
	}


	/* ---------------------------------------------
	 Progress Bar Animation
	 --------------------------------------------- */
	function initProgressBar(){
		$('.progress-bar > span').each(function(){
	        var $this = $(this);
	        var width = $(this).data('percent');
	        $this.css({
	            'transition' : 'width 1.5s'
	        });

	        setTimeout(function() {
	            $this.filter(':visible').waypoint(function(direction) {
	                if( direction === 'down' ) {
	                    $this.css('width', width + '%');
	                }
	            } , { offset: '100%' } );
	        }, 500);
	    });
	}


	/* ---------------------------------------------
	 All Carousels
	 --------------------------------------------- */
	function initCarousel(){
		$('.client-carousel').owlCarousel({
			pagination: false,
			navigation: false,
			autoPlay: true,
			slideSpeed : 500,
			items : 5,
			itemsDesktop : [991,3],
			itemsDesktopSmall : [768,3]
		});

		$('.reference-carousel').owlCarousel({
			pagination: true,
			navigation: false,
			autoPlay: true,
			slideSpeed : 500,
			items : 1
		});
	}


	/* ---------------------------------------------
	 Wow Animation
	 --------------------------------------------- */
	function initWowAnimation(){
		var wow = new WOW({
				boxClass: 'wow',
				animateClass: 'animated',
				offset: 150,
				mobile: false,
				live: true
			}
		);
		wow.init();
	}


	/* ---------------------------------------------
	 Miscellaneous
	 --------------------------------------------- */
	function initMiscellaneous(){
		$('.single-blog-navigation').children('a').addClass('btn-custom waves-effect');
		$('#wp-calendar').children('a').addClass('waves-effect waves-light');
		$('.pagination').children('.page-numbers').addClass('btn-circle waves-effect waves-light');
	}



	/* ---------------------------------------------
	 Contact Form
	 --------------------------------------------- */
	function initContactForm(){
		var receiver = $('#c-form').attr('data-email-address');

		var invalid_fields = [];

		function validateEmail(email){
			var regex = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
			return regex.test(email.toLowerCase());
		}

		$("#c-form").on('submit', function (e){
			e.preventDefault();

			$('#c-form-spinner').addClass('active');

			var name = $("#name").val();
			var email = $("#email").val();
			var subject = $("#subject").val();
			var message = $("#message").val();

			if(name.length < 3){
				invalid_fields.push('#name');
			}else{
				var name_index = invalid_fields.indexOf('#name');
				if(name_index > -1){
					invalid_fields.splice(name_index);
				}
			}
			if(email == '' || validateEmail(email) == false){
				invalid_fields.push('#email');
			}else{
				var email_index = invalid_fields.indexOf('#email');
				if(email_index > -1){
					invalid_fields.splice(email_index);
				}
			}
			if(subject.length < 3){
				invalid_fields.push('#subject');
			}else{
				var subject_index = invalid_fields.indexOf('#subject');
				if(subject_index > -1){
					invalid_fields.splice(subject_index);
				}
			}
			if(message.length < 3){
				invalid_fields.push('#message');
			}else{
				var message_index = invalid_fields.indexOf('#message');
				if(message_index > -1){
					invalid_fields.splice(message_index);
				}
			}

			if(invalid_fields.length > 0){
				$('.c-form-input').removeClass('invalid-field');
				for(var i = 0; i < invalid_fields.length; i++){
					$(invalid_fields[i]).addClass('invalid-field');
				}
				$('.c-form-alert').addClass('active');
				$('.alert-attention').addClass('active');
			}else{
				$.ajax({
					type: "POST",
					url: "cf-process.php",
					data: {
						'name': name,
						'email': email,
						'message': message,
						'subject': subject,
						'receiver': receiver
					},
					success : function(text){
						if (text === 'success'){
							$('.c-form-alert').addClass('active');
							$('.alert-success').addClass('active');
							$('#c-form')[0].reset();
							$('.c-form-input').removeClass('invalid-field');
						}else{
							$('.c-form-alert').addClass('active');
							$('.alert-error').addClass('active');
						}
					}
				});
			}
		});

		$('.alert-close a').on('click', function(){
			$('.alert-content').removeClass('active');
			$('.c-form-alert').removeClass('active');
			$('#c-form-spinner').removeClass('active');
		});
	}


	/* ---------------------------------------------
	 Contact Map Height
	 --------------------------------------------- */
	function initContactMapHeight(){
		$('.contact-map').height($('.contact-form').height());
	}


	/* ---------------------------------------------
	 Google Map
	 --------------------------------------------- */
	function initGoogleMap(){
		if(document.getElementById('g-map')){
			var latitude = document.getElementById('g-map').getAttribute('data-latitude'),
				longitude = document.getElementById('g-map').getAttribute('data-longitude');
		}else{
			return;
		}

		var mapStyles = [
			[{"featureType":"water","elementType":"geometry","stylers":[{"color":"#e9e9e9"},{"lightness":17}]},{"featureType":"landscape","elementType":"geometry","stylers":[{"color":"#f5f5f5"},{"lightness":20}]},{"featureType":"road.highway","elementType":"geometry.fill","stylers":[{"color":"#ffffff"},{"lightness":17}]},{"featureType":"road.highway","elementType":"geometry.stroke","stylers":[{"color":"#ffffff"},{"lightness":29},{"weight":0.2}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#ffffff"},{"lightness":18}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#ffffff"},{"lightness":16}]},{"featureType":"poi","elementType":"geometry","stylers":[{"color":"#f5f5f5"},{"lightness":21}]},{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#dedede"},{"lightness":21}]},{"elementType":"labels.text.stroke","stylers":[{"visibility":"on"},{"color":"#ffffff"},{"lightness":16}]},{"elementType":"labels.text.fill","stylers":[{"saturation":36},{"color":"#333333"},{"lightness":40}]},{"elementType":"labels.icon","stylers":[{"visibility":"off"}]},{"featureType":"transit","elementType":"geometry","stylers":[{"color":"#f2f2f2"},{"lightness":19}]},{"featureType":"administrative","elementType":"geometry.fill","stylers":[{"color":"#fefefe"},{"lightness":20}]},{"featureType":"administrative","elementType":"geometry.stroke","stylers":[{"color":"#fefefe"},{"lightness":17},{"weight":1.2}]}],
			[{"featureType":"administrative","elementType":"all","stylers":[{"saturation":"-100"}]},{"featureType":"administrative.province","elementType":"all","stylers":[{"visibility":"off"}]},{"featureType":"landscape","elementType":"all","stylers":[{"saturation":-100},{"lightness":65},{"visibility":"on"}]},{"featureType":"poi","elementType":"all","stylers":[{"saturation":-100},{"lightness":"50"},{"visibility":"simplified"}]},{"featureType":"road","elementType":"all","stylers":[{"saturation":"-100"}]},{"featureType":"road.highway","elementType":"all","stylers":[{"visibility":"simplified"}]},{"featureType":"road.arterial","elementType":"all","stylers":[{"lightness":"30"}]},{"featureType":"road.local","elementType":"all","stylers":[{"lightness":"40"}]},{"featureType":"transit","elementType":"all","stylers":[{"saturation":-100},{"visibility":"simplified"}]},{"featureType":"water","elementType":"geometry","stylers":[{"hue":"#ffff00"},{"lightness":-25},{"saturation":-97}]},{"featureType":"water","elementType":"labels","stylers":[{"lightness":-25},{"saturation":-100}]}],
			[{"featureType":"all","elementType":"labels.text.fill","stylers":[{"saturation":36},{"color":"#000000"},{"lightness":40}]},{"featureType":"all","elementType":"labels.text.stroke","stylers":[{"visibility":"on"},{"color":"#000000"},{"lightness":16}]},{"featureType":"all","elementType":"labels.icon","stylers":[{"visibility":"off"}]},{"featureType":"administrative","elementType":"geometry.fill","stylers":[{"color":"#000000"},{"lightness":20}]},{"featureType":"administrative","elementType":"geometry.stroke","stylers":[{"color":"#000000"},{"lightness":17},{"weight":1.2}]},{"featureType":"landscape","elementType":"geometry","stylers":[{"color":"#000000"},{"lightness":20}]},{"featureType":"poi","elementType":"geometry","stylers":[{"color":"#000000"},{"lightness":21}]},{"featureType":"road.highway","elementType":"geometry.fill","stylers":[{"color":"#000000"},{"lightness":17}]},{"featureType":"road.highway","elementType":"geometry.stroke","stylers":[{"color":"#000000"},{"lightness":29},{"weight":0.2}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#000000"},{"lightness":18}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#000000"},{"lightness":16}]},{"featureType":"transit","elementType":"geometry","stylers":[{"color":"#000000"},{"lightness":19}]},{"featureType":"water","elementType":"geometry","stylers":[{"color":"#000000"},{"lightness":17}]}],
			[{"featureType":"administrative","elementType":"labels.text.fill","stylers":[{"color":"#444444"}]},{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2f2f2"}]},{"featureType":"poi","elementType":"all","stylers":[{"visibility":"off"}]},{"featureType":"road","elementType":"all","stylers":[{"saturation":-100},{"lightness":45}]},{"featureType":"road.highway","elementType":"all","stylers":[{"visibility":"simplified"}]},{"featureType":"road.arterial","elementType":"labels.icon","stylers":[{"visibility":"off"}]},{"featureType":"transit","elementType":"all","stylers":[{"visibility":"off"}]},{"featureType":"water","elementType":"all","stylers":[{"color":"#46bcec"},{"visibility":"on"}]}],
			[{"featureType":"administrative.country","elementType":"geometry","stylers":[{"visibility":"simplified"},{"hue":"#ff0000"}]}],
			[{"featureType":"landscape","stylers":[{"hue":"#FFBB00"},{"saturation":43.400000000000006},{"lightness":37.599999999999994},{"gamma":1}]},{"featureType":"road.highway","stylers":[{"hue":"#FFC200"},{"saturation":-61.8},{"lightness":45.599999999999994},{"gamma":1}]},{"featureType":"road.arterial","stylers":[{"hue":"#FF0300"},{"saturation":-100},{"lightness":51.19999999999999},{"gamma":1}]},{"featureType":"road.local","stylers":[{"hue":"#FF0300"},{"saturation":-100},{"lightness":52},{"gamma":1}]},{"featureType":"water","stylers":[{"hue":"#0078FF"},{"saturation":-13.200000000000003},{"lightness":2.4000000000000057},{"gamma":1}]},{"featureType":"poi","stylers":[{"hue":"#00FF6A"},{"saturation":-1.0989010989011234},{"lightness":11.200000000000017},{"gamma":1}]}]
		];
		var mapStyleIndex = Number(document.getElementById('g-map').getAttribute('data-mapstyle'));

		var mapOptions = {
			scrollwheel: false,
			zoom: 14,
			center: new google.maps.LatLng(latitude, longitude),
			zoomControlOptions: {
				position: google.maps.ControlPosition.TOP_RIGHT
			},
			styles: mapStyles[mapStyleIndex-1]
		};

		var mapElement = document.getElementById('g-map');
		var map = new google.maps.Map(mapElement, mapOptions);
		var imageSrc = mapElement.getAttribute('data-map-marker');
		var image = new google.maps.MarkerImage(imageSrc,
			null, null, null, new google.maps.Size(48,48))

		var myLatLng = new google.maps.LatLng(latitude, longitude);
		var marker = new google.maps.Marker({
			position: myLatLng,
			map: map,
			icon: image
		});
	}

})(jQuery);
