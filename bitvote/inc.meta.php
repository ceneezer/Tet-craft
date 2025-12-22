<?php /*
Update html tag to include the itemscope and itemtype attributes.
$title < 60
$meta_desc <155 chrs
$meta_t_jpg 2:1 with minimum dimensions of 300x157 or maximum of 4096x4096 pixels. Images must be less than 5MB in size. JPG, PNG, WEBP and GIF
$meta_fb_jpg 1.91:1 ratio and minimum recommended dimensions of 1200x630 
$meta_gp_jpg ??
<meta name="twitter:creator" content="@author_handle">
<meta name="twitter:data1" content="$3">
<meta name="twitter:label1" content="Price">
<meta name="twitter:data2" content="Black">
<meta name="twitter:label2" content="Color">
<meta property="og:url" content="http://www.example.com/" />
<meta property="og:site_name" content="Site Name, i.e. Moz" />
<meta property="og:price:amount" content="15.00" />
<meta property="og:price:currency" content="USD" />
*/ ?>
   <title><?=$title?></title>
   <meta name="description" content="<?=$meta_desc?>" />
<!-- Twitter Card data -->
   <meta name="twitter:card" content="summary_large_image"><!-- product|app|player|summary|summary_large_image -->
   <meta name="twitter:site" content="@cenezer">
   <meta name="twitter:title" content="<?=$title?>">
   <meta name="twitter:description" content="<?=$meta_desc?>">
   <meta name="twitter:image" content="<?=$meta_t_jpg?>">
<!-- Open Graph data -->
   <meta property="og:title" content="<?=$title?>" />
   <meta property="og:type" content="<?=(isset($_REQUEST['page']))?'article':'website'?>" />
   <meta property="og:image" content="<?=$meta_fb_jpg?>" />
   <meta property="og:description" content="<?=$meta_desc?>" />
<!-- Schema.org markup for Google+ -->
   <meta itemprop="name" content="<?=$title?>">
   <meta itemprop="description" content="<?=$meta_desc?>">
   <meta itemprop="image" content="<?=$meta_gp_jpg?>">